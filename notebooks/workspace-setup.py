# Databricks notebook source
# MAGIC %sql
# MAGIC
# MAGIC CREATE CATALOG IF NOT EXISTS ps_video
# MAGIC CREATE SCHEMA IF NOT EXISTS ps_video.cv
# MAGIC CREATE VOLUME IF NOT EXISTS ps_video.cv.auto_segment