#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2026 fred <github-fred@hidzz.com>
#
# Distributed under terms of the BSD 3-Clause license.

__all__ = ["gen_filename", "fetch_and_filter", "render_html"]

from .wap import fetch_and_filter, gen_filename, render_html
