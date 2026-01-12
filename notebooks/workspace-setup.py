# Databricks notebook source
# MAGIC %sql
# MAGIC
# MAGIC CREATE CATALOG IF NOT EXISTS pubsec_video_processing;
# MAGIC CREATE SCHEMA IF NOT EXISTS pubsec_video_processing.cv;
# MAGIC CREATE VOLUME IF NOT EXISTS pubsec_video_processing.cv.auto_segment;