"""
title: Private Data Chat
author: lschiemanowski
version: 0.1.0
description: Clarify on synthetic data, confirm, run trusted DSA, and stop.
"""

from importlib import import_module

# Open WebUI rewrites legacy static imports beginning with ``apps`` in uploaded Function
# source. Dynamic import keeps this loader pointed at the separately mounted DSA application.
Pipe = import_module("apps.private_data_chat.openwebui_pipe").Pipe
