#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
事件日志 - 简化版
"""

class Logger:
    def info(self, msg):
        print(f"INFO: {msg}")
    
    def warning(self, msg):
        print(f"WARNING: {msg}")
    
    def error(self, msg):
        print(f"ERROR: {msg}")

LOGGER = Logger()
