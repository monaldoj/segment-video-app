import base64
import mimetypes
import os
import re
import time
from urllib.parse import urlencode
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from flask import Response, abort, request, stream_with_context


INPUTS_DIR = "/Volumes/justinm/cv/auto_segment/inputs"
OUTPUTS_DIR = "/Volumes/justinm/cv/auto_segment/outputs"
DESCRIPTIONS_DIR = OUTPUTS_DIR.rsplit("/", 1)[0] + "/descriptions"
JOB_ID = 774232501138119


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _sanitize_filename(filename: str) -> str:
    """
    Keep it simple and safe for filesystem paths.
    Also helps prevent path traversal.
    """
    filename = os.path.basename(filename)
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return filename or f"upload_{int(time.time())}.bin"


def _unique_name(filename: str) -> str:
    safe = _sanitize_filename(filename)
    stem, ext = os.path.splitext(safe)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{ts}{ext}"


def _parse_upload(contents: str) -> bytes:
    """
    Dash upload contents are: "data:<mime>;base64,<payload>"
    """
    if not contents or "," not in contents:
        raise ValueError("Upload payload is empty or malformed.")
    _, b64 = contents.split(",", 1)
    return base64.b64decode(b64)


def _get_client() -> WorkspaceClient:
    # In Databricks Apps, auth is typically auto-configured. For local dev, use env vars.
    return WorkspaceClient()


def _ensure_dir(w: WorkspaceClient, path: str) -> None:
    try:
        w.files.get_metadata(path)
    except NotFound:
        w.files.create_directory(path)


def _upload_to_volume(w: WorkspaceClient, volume_path: str, data: bytes) -> None:
    # files.upload expects binary content and writes into UC volumes paths like /Volumes/...
    w.files.upload(volume_path, contents=data, overwrite=True)


def _output_exists(w: WorkspaceClient, path: str) -> bool:
    try:
        w.files.get_metadata(path)
        return True
    except NotFound:
        return False


def _description_path_for_output(output_path: str) -> str:
    """
    Description defaults to the same filename as the output video, but:
    - directory: outputs -> descriptions
    - extension: .txt
    """
    base = os.path.basename(output_path)
    stem, _ext = os.path.splitext(base)
    return f"{DESCRIPTIONS_DIR}/{stem}.txt"


def _download_text(w: WorkspaceClient, path: str) -> Optional[str]:
    try:
        dl = w.files.download(path)
        try:
            raw = dl.contents.read()
        finally:
            try:
                dl.contents.close()
            except Exception:
                pass
        return raw.decode("utf-8", errors="replace")
    except NotFound:
        return None


def _download_url(w: WorkspaceClient, path: str) -> str:
    # The Databricks SDK Files API does not expose a presigned download URL helper.
    # Instead, this app serves a streaming download route backed by `w.files.download(...)`.
    qs = urlencode({"path": path, "v": str(int(time.time()))})
    # Use a relative URL so this keeps working even when the app is hosted behind a URL prefix.
    return f"download?{qs}"


def _start_run(
    w: WorkspaceClient,
    trigger_location: str,
    prompt: str,
    frame_stride: int = 5,
    truncate: bool = False,
) -> int:
    run = w.jobs.run_now(
        job_id=JOB_ID,
        # NOTE: notebook_params values are typically strings.
        notebook_params={
            "trigger_location": trigger_location,
            "prompt": prompt,
            "frame_stride": str(int(frame_stride)),
            "truncate": str(bool(truncate)).lower(),
        },
    )
    return int(run.run_id)


@dataclass(frozen=True)
class RunView:
    life_cycle_state: str
    result_state: Optional[str]
    state_message: Optional[str]
    start_time_ms: Optional[int]
    end_time_ms: Optional[int]


def _get_run_view(w: WorkspaceClient, run_id: int) -> RunView:
    r = w.jobs.get_run(run_id=run_id)
    st = r.state
    return RunView(
        life_cycle_state=str(st.life_cycle_state) if st else "UNKNOWN",
        result_state=str(st.result_state) if st and st.result_state else None,
        state_message=str(st.state_message) if st and st.state_message else None,
        start_time_ms=int(r.start_time) if r.start_time else None,
        end_time_ms=int(r.end_time) if r.end_time else None,
    )


def _elapsed_seconds(rv: RunView) -> Optional[int]:
    if not rv.start_time_ms:
        return None
    end_ms = rv.end_time_ms or int(_now_utc().timestamp() * 1000)
    return max(0, int((end_ms - rv.start_time_ms) / 1000))


def _fmt_elapsed(seconds: Optional[int]) -> str:
    if seconds is None:
        return "—"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m:02d}:{s:02d}"
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server


@server.get("/download")
def download_from_volume():
    """
    Stream a file from a Unity Catalog volume via Databricks SDK `files.download`.
    We intentionally restrict this to OUTPUTS_DIR to avoid arbitrary file reads.
    """
    path = request.args.get("path", "")
    if not path or not isinstance(path, str):
        abort(400, description="Missing 'path' query parameter.")
    if not path.startswith(OUTPUTS_DIR + "/"):
        abort(403, description="Only output volume paths are allowed.")

    w = _get_client()
    if not _output_exists(w, path):
        abort(404, description="File not found.")

    dl = w.files.download(path)
    content_stream = dl.contents

    def generate():
        try:
            while True:
                chunk = content_stream.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                content_stream.close()
            except Exception:
                pass

    mimetype, _ = mimetypes.guess_type(path)
    return Response(
        stream_with_context(generate()),
        mimetype=mimetype or "application/octet-stream",
        headers={
            "Content-Disposition": f'inline; filename="{os.path.basename(path)}"',
            "Cache-Control": "no-store",
        },
    )


app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("🎬 Find Objects in Videos"),
                        html.Div(
                            [
                                html.Div(
                                    "Upload a video (or enter a full volume path) and enter a prompt to find objects in the video. "
                                    "When it finishes, an edited version of the video will appear, the objects will be highlighted, and a description of the objects found will be generated."
                                )
                            ],
                            className="text-muted",
                        ),
                    ],
                    width=12,
                )
            ],
            className="mt-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div(
                                        id="input-controls",
                                        children=[
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Div("1) Upload video", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                                                            dcc.Loading(
                                                                id="loading-upload",
                                                                type="default",
                                                                children=[
                                                                    dcc.Upload(
                                                                        id="upload-video",
                                                                        children=html.Div(
                                                                            [
                                                                                html.Div(
                                                                                    [
                                                                                        "Drag & drop or ",
                                                                                        html.Span("browse", className="browse-link"),
                                                                                        " to upload",
                                                                                    ],
                                                                                    style={"marginBottom": "8px"}
                                                                                ),
                                                                                html.Small(
                                                                                    "Tip: large videos may not work via browser upload. Try loading from existing Volume location. ",
                                                                                    className="text-muted",
                                                                                ),
                                                                            ],
                                                                            style={
                                                                                "display": "flex",
                                                                                "flexDirection": "column",
                                                                                "justifyContent": "center",
                                                                                "alignItems": "center",
                                                                                "height": "100%",
                                                                            }
                                                                        ),
                                                                        style={
                                                                            "width": "100%",
                                                                            "height": "120px",
                                                                            "borderWidth": "2px",
                                                                            "borderStyle": "dashed",
                                                                            "borderRadius": "10px",
                                                                            "textAlign": "center",
                                                                            "display": "flex",
                                                                            "alignItems": "center",
                                                                            "justifyContent": "center",
                                                                        },
                                                                        multiple=False,
                                                                    ),
                                                            html.Div(
                                                                id="upload-status",
                                                                className="mt-2",
                                                            ),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Small("Or load existing file from volume:", className="text-muted"),
                                                            dbc.InputGroup(
                                                                [
                                                                    dbc.Input(
                                                                        id="existing-filename",
                                                                        placeholder="e.g. my_video.mp4 or /Volumes/catalog/schema/volume/file.mp4",
                                                                        type="text",
                                                                    ),
                                                                    dbc.Button(
                                                                        "Load",
                                                                        id="load-existing",
                                                                        color="secondary",
                                                                        outline=True,
                                                                    ),
                                                                ],
                                                                className="mt-2",
                                                            ),
                                                        ],
                                                        className="mt-3",
                                                    ),
                                                ],
                                                width=12,
                                            )
                                        ]
                                    ),
                                            html.Hr(),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Div("2) Prompt", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                                                            dbc.Textarea(
                                                                id="prompt",
                                                                placeholder="Describe objects to find in the video...",
                                                                style={"minHeight": "110px"},
                                                            ),
                                                        ],
                                                        width=12,
                                                    )
                                                ]
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label("Frame stride"),
                                                            dbc.Input(
                                                                id="frame-stride",
                                                                type="number",
                                                                min=1,
                                                                step=1,
                                                                value=5,
                                                            ),
                                                            html.Small(
                                                                "Process every Nth frame (higher = faster, less accurate).",
                                                                className="text-muted",
                                                            ),
                                                        ],
                                                        width=12,
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("Truncate output"),
                                                            dbc.RadioItems(
                                                                id="truncate",
                                                                options=[
                                                                    {"label": "False (full video)", "value": "false"},
                                                                    {"label": "True (only matches)", "value": "true"},
                                                                ],
                                                                value="true",
                                                                inline=True,
                                                            ),
                                                            html.Small(
                                                                "If true, only output segments where the prompt is found.",
                                                                className="text-muted",
                                                            ),
                                                        ],
                                                        width=12,
                                                        md=6,
                                                        className="mt-3 mt-md-0",
                                                    ),
                                                ],
                                                className="mt-3",
                                            ),
                                            html.Hr(),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Button(
                                                                "3) Run auto-segment job",
                                                                id="run-job",
                                                                color="primary",
                                                                className="me-2",
                                                            ),
                                                            dbc.Button(
                                                                "Reset",
                                                                id="reset",
                                                                color="secondary",
                                                                outline=True,
                                                            ),
                                                        ],
                                                        width=12,
                                                    )
                                                ]
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        id="run-status",
                                        className="mt-3",
                                    ),
                                ]
                            ),
                            className="shadow-sm",
                        ),
                        dcc.Store(id="store-upload-path"),
                        dcc.Store(id="store-output-path"),
                        dcc.Store(id="store-run-id"),
                        dcc.Interval(id="poll-interval", interval=2500, n_intervals=0, disabled=True),
                    ],
                    width=12,
                    lg=6,
                    className="mt-3",
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Output"),
                                    html.Div(
                                        id="output-video",
                                        className="mt-2",
                                    ),
                                    html.Div(
                                        id="output-description",
                                        className="mt-3",
                                    ),
                                    html.Hr(),
                                    html.H6("Lookup existing output"),
                                    html.Div(
                                        "If you come back later, enter a filename to load "
                                        "from the outputs volume.",
                                        className="text-muted",
                                    ),
                                    dbc.Input(
                                        id="lookup-filename",
                                        placeholder="e.g. my_video_20250101_120000.mp4",
                                        className="mt-2",
                                    ),
                                    dbc.Button(
                                        "Load output by filename",
                                        id="lookup-button",
                                        color="info",
                                        className="mt-2",
                                    ),
                                    html.Div(id="lookup-status", className="mt-2"),
                                ]
                            ),
                            className="shadow-sm",
                        )
                    ],
                    width=12,
                    lg=6,
                    className="mt-3",
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Hr(),
                        html.Small(
                            [
                                html.B("Inputs Volume: "),
                                INPUTS_DIR + "/",
                                html.Br(),
                                html.B("Outputs Volume: "),
                                OUTPUTS_DIR + "/",
                                html.Br(),
                                html.B("Job ID: "),
                                str(JOB_ID),
                            ],
                            className="text-muted",
                        ),
                    ],
                    width=12,
                    className="mt-2 mb-4",
                )
            ]
        ),
    ],
    fluid=True,
)


@app.callback(
    Output("store-upload-path", "data"),
    Output("store-output-path", "data"),
    Output("upload-status", "children"),
    Input("upload-video", "contents"),
    State("upload-video", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents: str, filename: str):
    if not contents or not filename:
        raise dash.exceptions.PreventUpdate

    w = _get_client()
    _ensure_dir(w, INPUTS_DIR)
    _ensure_dir(w, OUTPUTS_DIR)

    safe_name = _unique_name(filename)
    input_path = f"{INPUTS_DIR}/{safe_name}"
    output_path = f"{OUTPUTS_DIR}/{safe_name}"

    try:
        data = _parse_upload(contents)
        _upload_to_volume(w, input_path, data)
    except Exception as e:
        return None, None, dbc.Alert(f"Upload failed: {e}", color="danger")

    return (
        input_path,
        output_path,
        dbc.Alert(
            [
                html.Div([html.B("Uploaded: "), safe_name]),
                html.Div([html.B("Input path: "), input_path], className="small text-muted"),
            ],
            color="success",
        ),
    )


@app.callback(
    Output("store-upload-path", "data", allow_duplicate=True),
    Output("store-output-path", "data", allow_duplicate=True),
    Output("upload-status", "children", allow_duplicate=True),
    Input("load-existing", "n_clicks"),
    State("existing-filename", "value"),
    prevent_initial_call=True,
)
def handle_load_existing(n_clicks: int, filename: Optional[str]):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not filename or not filename.strip():
        return None, None, dbc.Alert("Enter a filename to load.", color="warning")

    w = _get_client()
    filename_stripped = filename.strip()
    
    # Check if user provided a full volume path
    if filename_stripped.startswith("/Volumes/"):
        input_path = filename_stripped
        # Extract just the filename for output
        basename = os.path.basename(input_path)
        output_path = f"{OUTPUTS_DIR}/{basename}"
        display_name = input_path
    else:
        # Use default inputs directory
        safe_name = _sanitize_filename(filename_stripped)
        input_path = f"{INPUTS_DIR}/{safe_name}"
        output_path = f"{OUTPUTS_DIR}/{safe_name}"
        display_name = safe_name

    # Check if the file exists in the volume
    if not _output_exists(w, input_path):
        return (
            None,
            None,
            dbc.Alert(
                [
                    html.Div("File not found."),
                    html.Div([html.B("Looked for: "), input_path], className="small text-muted"),
                ],
                color="danger",
            ),
        )

    return (
        input_path,
        output_path,
        dbc.Alert(
            [
                html.Div([html.B("Loaded existing: "), display_name]),
                html.Div([html.B("Input path: "), input_path], className="small text-muted"),
            ],
            color="success",
        ),
    )


@app.callback(
    Output("store-run-id", "data"),
    Output("poll-interval", "disabled"),
    Output("run-status", "children"),
    Output("input-controls", "style"),
    Input("run-job", "n_clicks"),
    State("store-upload-path", "data"),
    State("prompt", "value"),
    State("frame-stride", "value"),
    State("truncate", "value"),
    prevent_initial_call=True,
)
def start_job(
    n_clicks: int,
    upload_path: Optional[str],
    prompt: Optional[str],
    frame_stride: Optional[int],
    truncate: Optional[str],
):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not upload_path:
        return None, True, dbc.Alert("Upload a video first.", color="warning"), {}
    if not prompt or not prompt.strip():
        return None, True, dbc.Alert("Enter a prompt first.", color="warning"), {}

    try:
        fs = int(frame_stride) if frame_stride is not None else 5
    except Exception:
        return None, True, dbc.Alert("Frame stride must be a positive integer.", color="warning"), {}
    if fs < 1:
        return None, True, dbc.Alert("Frame stride must be a positive integer.", color="warning"), {}

    tr = str(truncate).lower() == "true" if truncate is not None else False

    w = _get_client()
    try:
        run_id = _start_run(
            w,
            trigger_location=upload_path,
            prompt=prompt.strip(),
            frame_stride=fs,
            truncate=tr,
        )
    except Exception as e:
        return None, True, dbc.Alert(f"Failed to start job: {e}", color="danger"), {}

    return (
        run_id,
        False,
        dbc.Alert(
            [html.Div([html.B("Started run_id: "), str(run_id)]), html.Div("Polling job status...")],
            color="primary",
        ),
        {"pointerEvents": "none", "opacity": 0.5},
    )


@app.callback(
    Output("run-status", "children", allow_duplicate=True),
    Output("poll-interval", "disabled", allow_duplicate=True),
    Output("output-video", "children"),
    Output("output-description", "children"),
    Output("input-controls", "style", allow_duplicate=True),
    Input("poll-interval", "n_intervals"),
    State("store-run-id", "data"),
    State("store-output-path", "data"),
    prevent_initial_call=True,
)
def poll_run(n: int, run_id: Optional[int], output_path: Optional[str]):
    if not run_id:
        raise dash.exceptions.PreventUpdate

    w = _get_client()
    try:
        rv = _get_run_view(w, int(run_id))
    except Exception as e:
        return dbc.Alert(f"Error polling run: {e}", color="danger"), True, dash.no_update, dash.no_update, {}

    elapsed = _fmt_elapsed(_elapsed_seconds(rv))

    status_bits = [
        html.Div([html.B("Run: "), str(run_id)]),
        html.Div([html.B("State: "), rv.life_cycle_state]),
        html.Div([html.B("Elapsed: "), elapsed]),
    ]
    if rv.result_state:
        status_bits.append(html.Div([html.B("Result: "), rv.result_state]))
    if rv.state_message:
        status_bits.append(html.Div([html.B("Message: "), rv.state_message], className="small text-muted"))

    # If job terminated, stop polling job status and begin checking for output existence.
    # if rv.life_cycle_state == "TERMINATED":
    if rv.result_state:
        # Ungray input controls when job terminates (any result state)
        controls_style = {}
        
        if output_path and _output_exists(w, output_path):
            url = _download_url(w, output_path)
            video = html.Video(src=url, controls=True, style={"width": "100%", "maxHeight": "520px"})
            desc_path = _description_path_for_output(output_path)
            desc_txt = _download_text(w, desc_path)
            desc = (
                dbc.Alert(
                    [html.B("AI-Generated Description:"), html.Br(), dcc.Markdown(desc_txt or "No description found yet.")],
                    color="light",
                )
                if desc_txt is not None
                else html.Div(
                    ["AI-Generated Description: ", html.Span("No description found yet.", className="text-muted")]
                )
            )
            return dbc.Alert(status_bits, color="success"), True, video, desc, controls_style

        # Job finished but output not present yet. Keep polling a bit longer.
        waiting = dbc.Alert(
            status_bits
            + [
                html.Hr(),
                html.Div("Job finished; waiting for output file to appear..."),
                html.Div(output_path or "—", className="small text-muted"),
            ],
            color="warning",
        )
        return waiting, False, dash.no_update, dash.no_update, controls_style

    # Still running - keep grayed out
    return dbc.Alert(status_bits, color="primary"), False, dash.no_update, dash.no_update, {"pointerEvents": "none", "opacity": 0.5}


@app.callback(
    Output("output-video", "children", allow_duplicate=True),
    Output("output-description", "children", allow_duplicate=True),
    Output("lookup-status", "children"),
    Input("lookup-button", "n_clicks"),
    State("lookup-filename", "value"),
    prevent_initial_call=True,
)
def lookup_output(n_clicks: int, filename: Optional[str]):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not filename or not filename.strip():
        return dash.no_update, dash.no_update, dbc.Alert("Enter a filename to look up.", color="warning")

    safe = _sanitize_filename(filename.strip())
    output_path = f"{OUTPUTS_DIR}/{safe}"

    w = _get_client()
    try:
        if not _output_exists(w, output_path):
            return (
                dash.no_update,
                dash.no_update,
                dbc.Alert([html.Div("Not found:"), html.Div(output_path, className="small")], color="warning"),
            )

        url = _download_url(w, output_path)
        video = html.Video(src=url, controls=True, style={"width": "100%", "maxHeight": "520px"})
        desc_path = _description_path_for_output(output_path)
        desc_txt = _download_text(w, desc_path)
        desc = (
            dbc.Alert(
                [html.B("AI-Generated Description:"), html.Br(), dcc.Markdown(desc_txt or "No description found yet.")],
                color="light",
            )
            if desc_txt is not None
            else html.Div(["AI-Generated Description: ", html.Span("No description found yet.", className="text-muted")])
        )
        return (
            video,
            desc,
            dbc.Alert([html.Div([html.B("Loaded: "), safe])], color="success"),
        )
    except Exception as e:
        return dash.no_update, dash.no_update, dbc.Alert(f"Lookup failed: {e}", color="danger")


@app.callback(
    Output("store-upload-path", "data", allow_duplicate=True),
    Output("store-output-path", "data", allow_duplicate=True),
    Output("store-run-id", "data", allow_duplicate=True),
    Output("upload-status", "children", allow_duplicate=True),
    Output("run-status", "children", allow_duplicate=True),
    Output("output-video", "children", allow_duplicate=True),
    Output("output-description", "children", allow_duplicate=True),
    Output("lookup-status", "children", allow_duplicate=True),
    Output("prompt", "value", allow_duplicate=True),
    Output("frame-stride", "value", allow_duplicate=True),
    Output("truncate", "value", allow_duplicate=True),
    Output("poll-interval", "disabled", allow_duplicate=True),
    Output("input-controls", "style", allow_duplicate=True),
    Output("existing-filename", "value", allow_duplicate=True),
    Input("reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_all(n_clicks: int):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return None, None, None, None, None, None, None, None, "", 5, "false", True, {}, ""


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8050"))
    app.run_server(host=host, port=port, debug=False)


