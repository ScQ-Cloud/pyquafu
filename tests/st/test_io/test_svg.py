# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# wITHOUT wARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test svg."""
from mindquantum import qft, BarrierGate, RX

def test_svg():
    """
    test
    Description:
    Expectation:
    """
    text = (qft(range(3))+RX({'a':1.2}).on(1)+BarrierGate()).measure_all().svg()._repr_svg_()
    text_exp = """<svg xmlns="http://www.w3.org/2000/svg" width="710.64" height="188.0">\n<text x="20" y="30.0" font-size="16px" dominant-baseline="middle" text-anchor="start" font-family="Arial" >\nq0:\n </text>\n<text x="20" y="90.0" font-size="16px" dominant-baseline="middle" text-anchor="start" font-family="Arial" >\nq1:\n </text>\n<text x="20" y="150.0" font-size="16px" dominant-baseline="middle" text-anchor="start" font-family="Arial" >\nq2:\n </text>\n<line x1="48.8" x2="696.8" y1="30.0" y2="30.0" stroke="#adb0b8" stroke-width="1" />\n<line x1="48.8" x2="696.8" y1="90.0" y2="90.0" stroke="#adb0b8" stroke-width="1" />\n<line x1="48.8" x2="696.8" y1="150.0" y2="150.0" stroke="#adb0b8" stroke-width="1" />\n\n<rect x="72.8" y="10" width="40.0" height="40" rx="4" ry="4" fill="#5e7ce0" />\n<text x="92.8" y="30.0" font-size="20px" dominant-baseline="middle" text-anchor="middle" font-family="Arial" fill="#ffffff" >\nH\n </text>\n\n<circle cx="152.8" cy="90.0" r="4" fill="#fac209" />\n<line x1="152.8" x2="152.8" y1="30.0" y2="90.0" stroke="#fac209" stroke-width="3" />\n<rect x="132.8" y="10" width="40.0" height="40" rx="4" ry="4" fill="#fac209" />\n<text x="152.8" y="26.0" font-size="20px" dominant-baseline="middle" text-anchor="middle" font-family="Arial" fill="#ffffff" >\nPS\n </text>\n<text x="152.8" y="42.0" font-size="14.0px" dominant-baseline="middle" text-anchor="middle" font-family="Arial" fill="#ffffff" >\nπ/2\n </text>\n\n<circle cx="212.8" cy="150.0" r="4" fill="#fac209" />\n<line x1="212.8" x2="212.8" y1="30.0" y2="150.0" stroke="#fac209" stroke-width="3" />\n<rect x="192.8" y="10" width="40.0" height="40" rx="4" ry="4" fill="#fac209" />\n<text x="212.8" y="26.0" font-size="20px" dominant-baseline="middle" text-anchor="middle" font-family="Arial" fill="#ffffff" >\nPS\n </text>\n<text x="212.8" y="42.0" font-size="14.0px" dominant-baseline="middle" text-anchor="middle" font-family="Arial" fill="#ffffff" >\nπ/4\n </text>\n\n\n<rect x="252.8" y="70" width="40.0" height="40" rx="4" ry="4" fill="#5e7ce0" />\n<text x="272.8" y="90.0" font-size="20px" dominant-baseline="middle" text-anchor="middle" font-family="Arial" fill="#ffffff" >\nH\n </text>\n\n<circle cx="332.8" cy="150.0" r="4" fill="#fac209" />\n<line x1="332.8" x2="332.8" y1="90.0" y2="150.0" stroke="#fac209" stroke-width="3" />\n<rect x="312.8" y="70" width="40.0" height="40" rx="4" ry="4" fill="#fac209" />\n<text x="332.8" y="86.0" font-size="20px" dominant-baseline="middle" text-anchor="middle" font-family="Arial" fill="#ffffff" >\nPS\n </text>\n<text x="332.8" y="102.0" font-size="14.0px" dominant-baseline="middle" text-anchor="middle" font-family="Arial" fill="#ffffff" >\nπ/2\n </text>\n\n\n<rect x="372.8" y="130" width="40.0" height="40" rx="4" ry="4" fill="#5e7ce0" />\n<text x="392.8" y="150.0" font-size="20px" dominant-baseline="middle" text-anchor="middle" font-family="Arial" fill="#ffffff" >\nH\n </text>\n\n<line x1="452.8" x2="452.8" y1="10" y2="170" stroke-width="3" stroke="#16acff" />\n\n<rect x="432.8" y="10" width="40" height="40" rx="4" ry="4" fill="#16acff" />\n<path d="M 443.2 26.31384387633061 L 448.0 18.0 L 452.8 26.31384387633061 L 449.44 26.31384387633061 L 449.44 42.0 L 446.56 42.0 L 446.56 26.31384387633061 Z" fill="#ffffff" />\n<path d="M 462.40000000000003 33.68615612366939 L 457.6 42.0 L 452.8 33.68615612366939 L 456.16 33.68615612366939 L 456.16 18.0 L 459.04 18.0 L 459.04 33.68615612366939 Z" fill="#ffffff" />\n<rect x="432.8" y="130" width="40" height="40" rx="4" ry="4" fill="#16acff" />\n<path d="M 443.2 146.31384387633062 L 448.0 138.0 L 452.8 146.31384387633062 L 449.44 146.31384387633062 L 449.44 162.0 L 446.56 162.0 L 446.56 146.31384387633062 Z" fill="#ffffff" />\n<path d="M 462.40000000000003 153.68615612366938 L 457.6 162.0 L 452.8 153.68615612366938 L 456.16 153.68615612366938 L 456.16 138.0 L 459.04 138.0 L 459.04 153.68615612366938 Z" fill="#ffffff" />\n\n<rect x="492.8" y="70" width="80.0" height="40" rx="4" ry="4" fill="#fac209" />\n<text x="532.8" y="86.0" font-size="20px" dominant-baseline="middle" text-anchor="middle" font-family="Arial" fill="#ffffff" >\nRX\n </text>\n<text x="532.8" y="102.0" font-size="14.0px" dominant-baseline="middle" text-anchor="middle" font-family="Arial" fill="#ffffff" >\n1.2*a\n </text>\n\n<rect x="592.8" y="10" width="20" height="160" fill="gray" fill-opacity="0.8" />\n<rect x="632.8" y="10" width="40" height="40" rx="4" ry="4" fill="#ff7272" />\n<circle cx="652.8" cy="42.0" r="2" fill="#ffffff" />\n<path d="M 636.8 42.0 A 16.0 16.0 0 0 1 668.8 42.0" stroke="#ffffff" stroke-width="3" fill-opacity="0" />\n<path d="M 657.9273103968574 21.41923788646684 L 668.3196152422706 15.419237886466838 L 668.3196152422706 27.41923788646684 L 664.1626933041053 25.01923788646684 L 654.358845726812 42.0 L 652.2803847577293 40.8 L 662.0842323350226 23.81923788646684 Z" fill="#ffffff" />\n<rect x="632.8" y="70" width="40" height="40" rx="4" ry="4" fill="#ff7272" />\n<circle cx="652.8" cy="102.0" r="2" fill="#ffffff" />\n<path d="M 636.8 102.0 A 16.0 16.0 0 0 1 668.8 102.0" stroke="#ffffff" stroke-width="3" fill-opacity="0" />\n<path d="M 657.9273103968574 81.41923788646685 L 668.3196152422706 75.41923788646685 L 668.3196152422706 87.41923788646685 L 664.1626933041053 85.01923788646684 L 654.358845726812 102.0 L 652.2803847577293 100.8 L 662.0842323350226 83.81923788646684 Z" fill="#ffffff" />\n<rect x="632.8" y="130" width="40" height="40" rx="4" ry="4" fill="#ff7272" />\n<circle cx="652.8" cy="162.0" r="2" fill="#ffffff" />\n<path d="M 636.8 162.0 A 16.0 16.0 0 0 1 668.8 162.0" stroke="#ffffff" stroke-width="3" fill-opacity="0" />\n<path d="M 657.9273103968574 141.41923788646685 L 668.3196152422706 135.41923788646685 L 668.3196152422706 147.41923788646685 L 664.1626933041053 145.01923788646684 L 654.358845726812 162.0 L 652.2803847577293 160.8 L 662.0842323350226 143.81923788646685 Z" fill="#ffffff" />\n</svg>"""
    assert text == text_exp