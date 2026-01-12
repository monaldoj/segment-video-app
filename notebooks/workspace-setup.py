# Databricks notebook source
# MAGIC %sql
# MAGIC
# MAGIC CREATE CATALOG IF NOT EXISTS pubsec_video
# MAGIC CREATE SCHEMA IF NOT EXISTS pubsec_video.cv
# MAGIC CREATE VOLUME IF NOT EXISTS pubsec_video.cv.auto_segment