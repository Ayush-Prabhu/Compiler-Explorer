import sys
import re
import subprocess
import os
import glob
import tempfile
import difflib
import shutil
import math

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QFileDialog,
    QTextEdit, QPlainTextEdit, QMessageBox, QLabel, QListWidget,
    QSplitter, QGraphicsView, QVBoxLayout, QWidget, QTabWidget, QHBoxLayout,
    QFrame, QDialog, QLineEdit, QPushButton, QGridLayout, QGraphicsScene,
    QFormLayout, QFontComboBox, QSpinBox, QDialogButtonBox, QStyleFactory,
    QMenu, QToolBar, QComboBox, QSizePolicy
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtGui import (
    QSyntaxHighlighter, QTextCharFormat, QColor, QFont, QPainter, 
    QTextFormat, QTextCursor, QPalette
)
from PyQt5.QtCore import Qt, QUrl, QRect, QSize, QPoint, pyqtSignal
from PyQt5.QtSvg import QGraphicsSvgItem

# ========================================================
# 1. GRAPH ALGORITHMS (Lengauer-Tarjan)
# ========================================================

class Graph:
    def __init__(self):
        self.nodes = set()
        self.succs = {} # Map node -> list of successors
        self.preds = {} # Map node -> list of predecessors
        self.labels = {} # Map node -> text label (e.g. content)

    def add_node(self, node, label=""):
        self.nodes.add(node)
        if node not in self.succs: self.succs[node] = []
        if node not in self.preds: self.preds[node] = []
        if label: self.labels[node] = label

    def add_edge(self, u, v):
        self.succs[u].append(v)
        self.preds[v].append(u)

# --- INSERT IN SECTION 2: CFG PARSING & HELPERS ---
def parse_include_tree(stderr_output):
    """ Parses GCC -H output into a Graph object """
    g = Graph()
    g.add_node("ROOT", "Project Root")
    
    # Stack stores [ (depth, node_name) ]
    stack = [(-1, "ROOT")]
    
    # Regex to capture dots and filename: "... /usr/include/stdio.h"
    regex = re.compile(r'^(\.+)\s+(.*)')
    
    for line in stderr_output.splitlines():
        match = regex.match(line)
        if match:
            dots = match.group(1)
            filepath = match.group(2)
            filename = os.path.basename(filepath)
            depth = len(dots)
            
            # Create unique ID to handle same header included multiple times
            node_id = f"{filename}_{id(line)}" 
            g.add_node(node_id, filename)
            
            # Find parent in stack
            while stack and stack[-1][0] >= depth:
                stack.pop()
            
            parent_id = stack[-1][1]
            g.add_edge(parent_id, node_id)
            
            stack.append((depth, node_id))
            
    return g

def parse_cfg_to_graph(cfg_text):
    """ Parses GCC CFG dump into a Graph object """
    g = Graph()
    blocks = {}
    current_block = None
    block_lines = []
    
    lines = cfg_text.splitlines()
    for line in lines:
        line = line.strip()
        if re.match(r'^<bb \d+>', line):
            if current_block is not None:
                blocks[current_block] = "\n".join(block_lines).strip()
                block_lines = []
            current_block = re.findall(r'<bb (\d+)>', line)[0]
            g.add_node(current_block)
        elif line.startswith(";;"):
            pass 
        elif current_block is not None:
            block_lines.append(line)
            
    if current_block and block_lines:
        blocks[current_block] = "\n".join(block_lines).strip()
        
    for node, content in blocks.items():
        g.labels[node] = content

    for line in lines:
        if line.startswith(";;"):
            succ_match = re.match(r';;\s*([0-9]+)\s+succs\s+\{(.+?)\}', line)
            if succ_match:
                src = succ_match.group(1)
                targets = re.findall(r'\d+', succ_match.group(2))
                for tgt in targets:
                    g.add_node(tgt) # Ensure target exists
                    g.add_edge(src, tgt)
    return g

def compute_dominators(g, entry_node):
    if entry_node not in g.nodes:
        return {}

    parent = {}
    vertex = [None] 
    semi = {}      
    label = {}     
    ancestor = {}
    bucket = {n: set() for n in g.nodes}
    dfnum = {} 
    n = 0
    
    def dfs(u):
        nonlocal n
        n += 1
        dfnum[u] = n
        vertex.append(u)
        label[u] = u
        semi[u] = u
        ancestor[u] = None
        
        for v in g.succs.get(u, []):
            if v not in dfnum:
                parent[v] = u
                dfs(v)
    
    dfs(entry_node)
    
    def compress(v):
        if ancestor[ancestor[v]] is not None:
            compress(ancestor[v])
            if dfnum[semi[label[ancestor[v]]]] < dfnum[semi[label[v]]]:
                label[v] = label[ancestor[v]]
            ancestor[v] = ancestor[ancestor[v]]

    def eval_node(v):
        if ancestor[v] is None: return v
        compress(v)
        return label[v]

    dom = {} 

    for i in range(n, 1, -1):
        w = vertex[i] 
        for v in g.preds.get(w, []):
            if v in dfnum: 
                u = eval_node(v)
                if dfnum[semi[u]] < dfnum[semi[w]]:
                    semi[w] = semi[u]
        bucket[semi[w]].add(w)
        link_node = parent[w]
        ancestor[w] = link_node
        for v in bucket[link_node]:
            u = eval_node(v)
            dom[v] = u if semi[u] == semi[v] else link_node
        bucket[link_node].clear()

    for i in range(2, n + 1):
        w = vertex[i]
        if w != dom[w]: dom[w] = dom[dom[w]]

    dom[entry_node] = None
    return dom

def generate_dom_tree_dot(idom_map, graph_labels):
    dot_lines = ["digraph DominatorTree {", "node [shape=box, fontname=\"Courier\"];"]
    for node, label in graph_labels.items():
        clean_label = label.replace("\"", "\\\"")[:30] + "..." if len(label) > 30 else label.replace("\"", "\\\"")
        dot_lines.append(f'"{node}" [label="{node}\\n{clean_label}"];')
    for u, v in idom_map.items():
        if v is not None:
            dot_lines.append(f'"{v}" -> "{u}";')
    dot_lines.append("}")
    return "\n".join(dot_lines)

# ========================================================
# 2. CFG PARSING & HELPERS
# ========================================================

def parse_cfg_to_dot(cfg_text):
    dot_lines = ["digraph CFG {", "node [shape=box, fontname=\"Courier\"];"]
    g = parse_cfg_to_graph(cfg_text)
    for node, content in g.labels.items():
        label = content.replace("\"", "\\\"")
        dot_lines.append(f'"{node}" [label="{node}:\\n{label}"];')
    for src, targets in g.succs.items():
        for tgt in targets:
            dot_lines.append(f'"{src}" -> "{tgt}";')
    dot_lines.append("}")
    return "\n".join(dot_lines)

def extract_cfgs_per_function(cfg_text):
    functions = {}
    current_func = None
    current_lines = []
    for line in cfg_text.splitlines():
        if line.startswith(";; Function "):
            if current_func and current_lines:
                functions[current_func] = "\n".join(current_lines)
            current_func = re.findall(r";; Function (\w+)", line)[0]
            current_lines = []
        elif current_func is not None:
            current_lines.append(line)
    if current_func and current_lines:
        functions[current_func] = "\n".join(current_lines)
    return functions

def generate_pass_diffs(passes):
    diffs = []
    for i in range(len(passes) - 1):
        _, name1, file1 = passes[i]
        _, name2, file2 = passes[i + 1]
        with open(file1) as f1, open(file2) as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
        diff = list(difflib.unified_diff(lines1, lines2, lineterm=""))
        diffs.append(((name1, name2), diff))
    return diffs

def parse_assembly_mapping(asm_content):
    """ 
    Parses GCC -fverbose-asm output to map source lines to assembly lines.
    Returns: dict { source_line_num: [ (asm_line_start, asm_line_end), ... ] }
    """
    mapping = {}
    lines = asm_content.splitlines()
    
    current_source_line = -1
    block_start = -1
    
    # Regex to match lines like: # filename.c:12: ...
    source_marker = re.compile(r'^\s*#\s+.*:(\d+):')
    
    for i, line in enumerate(lines):
        match = source_marker.search(line)
        if match:
            new_line = int(match.group(1))
            
            if current_source_line != -1 and block_start != -1:
                if current_source_line not in mapping:
                    mapping[current_source_line] = []
                mapping[current_source_line].append((block_start, i - 1))
            
            current_source_line = new_line
            block_start = i
    
    if current_source_line != -1 and block_start != -1:
        if current_source_line not in mapping:
            mapping[current_source_line] = []
        mapping[current_source_line].append((block_start, len(lines) - 1))
        
    return mapping

def analyze_asm_stats(asm_content):
    """ Returns a string summary of instruction mix """
    lines = asm_content.splitlines()
    counts = {
        "Total Ops": 0,
        "Memory (mov/push/pop)": 0,
        "Math (add/sub/mul)": 0,
        "Branch (jmp/jxx/call)": 0,
        "Labels": 0
    }
    
    # Basic heuristics
    for line in lines:
        line = line.strip()
        if not line or line.startswith((".", "#", "/")): continue
        if line.endswith(":"): 
            counts["Labels"] += 1
            continue
            
        # Assuming instruction if we are here
        counts["Total Ops"] += 1
        parts = line.split()
        if not parts: continue
        mnemonic = parts[0].lower()
        
        if mnemonic.startswith(('mov', 'push', 'pop', 'lea')):
            counts["Memory (mov/push/pop)"] += 1
        elif mnemonic.startswith(('add', 'sub', 'imul', 'idiv', 'inc', 'dec', 'xor', 'and', 'or', 'shl', 'shr')):
            counts["Math (add/sub/mul)"] += 1
        elif mnemonic.startswith(('j', 'call', 'ret')):
            counts["Branch (jmp/jxx/call)"] += 1
            
    summary = " // --- ASSEMBLY STATS ---\n"
    for k, v in counts.items():
        summary += f" // {k}: {v}\n"
    summary += " // ----------------------\n\n"
    return summary

# ========================================================
# 3. CUSTOM WIDGETS
# ========================================================

class LogViewer(QPlainTextEdit):
    """ Specialized Viewer for Build/Sanitizer output. Double-click jumps to code. """
    navigation_requested = pyqtSignal(str, int)

    def __init__(self, parent=None, font=None):
        super().__init__(parent)
        self.setReadOnly(True)
        if font:
            self.setFont(font)
        else:
            self.setFont(QFont("Courier", 10))

    def mouseDoubleClickEvent(self, event):
        cursor = self.cursorForPosition(event.pos())
        cursor.select(QTextCursor.LineUnderCursor)
        line_text = cursor.selectedText()
        
        # Regex to find "filename.c:123" or "filename.c:123:45"
        match = re.search(r'([a-zA-Z0-9_\-]+\.(?:c|cpp|cc|cxx|h|hpp)):(\d+)', line_text)
        if match:
            filename = match.group(1)
            line = int(match.group(2))
            self.navigation_requested.emit(filename, line)
        
        super().mouseDoubleClickEvent(event)

class PreferencesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Preferences")
        self.setFixedWidth(450)
        
        layout = QFormLayout()
        
        self.compiler_input = QLineEdit(self.parent_window.compiler_path)
        layout.addRow("C Compiler Path:", self.compiler_input)

        self.cpp_compiler_input = QLineEdit(self.parent_window.cpp_compiler_path)
        layout.addRow("C++ Compiler Path:", self.cpp_compiler_input)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["GCC", "Clang"])
        self.type_combo.setCurrentText(self.parent_window.compiler_type)
        layout.addRow("Compiler Backend Type:", self.type_combo)
        
        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(self.parent_window.editor_font)
        layout.addRow("Editor Font:", self.font_combo)
        
        self.size_spin = QSpinBox()
        self.size_spin.setRange(6, 72)
        self.size_spin.setValue(self.parent_window.editor_font.pointSize())
        layout.addRow("Font Size:", self.size_spin)
        
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.apply_settings)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)
        self.setLayout(layout)

    def apply_settings(self):
        self.parent_window.compiler_path = self.compiler_input.text()
        self.parent_window.cpp_compiler_path = self.cpp_compiler_input.text()
        self.parent_window.compiler_type = self.type_combo.currentText()
        font = self.font_combo.currentFont()
        font.setPointSize(self.size_spin.value())
        self.parent_window.update_editor_font(font)
        self.accept()

class FindReplaceDialog(QDialog):
    def __init__(self, editor, parent=None):
        super().__init__(parent)
        self.editor = editor
        self.setWindowTitle("Find & Replace")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.resize(300, 150)
        layout = QGridLayout()
        layout.addWidget(QLabel("Find:"), 0, 0)
        self.find_input = QLineEdit()
        layout.addWidget(self.find_input, 0, 1)
        layout.addWidget(QLabel("Replace:"), 1, 0)
        self.replace_input = QLineEdit()
        layout.addWidget(self.replace_input, 1, 1)
        self.btn_find = QPushButton("Find Next")
        self.btn_find.clicked.connect(self.find_next)
        layout.addWidget(self.btn_find, 2, 0)
        self.btn_replace = QPushButton("Replace")
        self.btn_replace.clicked.connect(self.replace)
        layout.addWidget(self.btn_replace, 2, 1)
        self.btn_replace_all = QPushButton("Replace All")
        self.btn_replace_all.clicked.connect(self.replace_all)
        layout.addWidget(self.btn_replace_all, 3, 0, 1, 2)
        self.setLayout(layout)

    def find_next(self):
        text = self.find_input.text()
        if not text: return
        found = self.editor.find(text)
        if not found:
            cursor = self.editor.textCursor()
            cursor.movePosition(QTextCursor.Start)
            self.editor.setTextCursor(cursor)
            found = self.editor.find(text)
            if not found: QMessageBox.information(self, "Find", "Text not found.")

    def replace(self):
        cursor = self.editor.textCursor()
        if cursor.hasSelection() and cursor.selectedText() == self.find_input.text():
            cursor.insertText(self.replace_input.text())
            self.find_next()
        else: self.find_next()

    def replace_all(self):
        text = self.find_input.text()
        replace_text = self.replace_input.text()
        if not text: return
        cursor = self.editor.textCursor()
        cursor.beginEditBlock()
        cursor.movePosition(QTextCursor.Start)
        self.editor.setTextCursor(cursor)
        count = 0
        while self.editor.find(text):
            cursor = self.editor.textCursor()
            cursor.insertText(replace_text)
            count += 1
        cursor.endEditBlock()
        QMessageBox.information(self, "Replace All", f"Replaced {count} occurrences.")

# --- Highlighters ---

class BaseHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.rules = []
        self.dark_mode = False
    def set_theme(self, is_dark):
        self.dark_mode = is_dark
        self.update_rules()
        self.rehighlight()
    def update_rules(self): pass
    def highlightBlock(self, text):
        for pattern, fmt in self.rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), fmt)

class CppSyntaxHighlighter(BaseHighlighter):
    def update_rules(self):
        self.rules = []
        c_keyword = QColor("cyan") if self.dark_mode else QColor("blue")
        c_prep = QColor("orange") if self.dark_mode else QColor("darkred")
        c_comment = QColor("lightgreen") if self.dark_mode else QColor("green")
        c_string = QColor("#ff79c6") if self.dark_mode else QColor("magenta")
        fmt_kw = QTextCharFormat(); fmt_kw.setForeground(c_keyword); fmt_kw.setFontWeight(QFont.Bold)
        fmt_prep = QTextCharFormat(); fmt_prep.setForeground(c_prep)
        fmt_com = QTextCharFormat(); fmt_com.setForeground(c_comment)
        fmt_str = QTextCharFormat(); fmt_str.setForeground(c_string)
        
        keywords = [
            "int", "float", "char", "double", "void", "return", "if", "else", "while", 
            "for", "do", "switch", "case", "break", "continue", "struct", "typedef", 
            "static", "const", "sizeof", "long", "short", "unsigned", "signed", "volatile",
            "class", "namespace", "template", "typename", "public", "private", "protected",
            "virtual", "override", "final", "new", "delete", "this", "friend", "using",
            "auto", "nullptr", "bool", "true", "false", "try", "catch", "throw", "constexpr"
        ]
        for word in keywords: self.rules.append((re.compile(r'\b' + word + r'\b'), fmt_kw))
        self.rules.append((re.compile(r'^\s*#\s*(include|define|ifdef|ifndef|endif|pragma).*$'), fmt_prep))
        self.rules.append((re.compile(r'//.*'), fmt_com))
        self.rules.append((re.compile(r'/\*.*?\*/', re.DOTALL), fmt_com))
        self.rules.append((re.compile(r'"[^"\\]*(\\.[^"\\]*)*"'), fmt_str))

class AssemblyHighlighter(BaseHighlighter):
    def update_rules(self):
        self.rules = []
        c_reg = QColor("#ff5555") if self.dark_mode else QColor("#a31515")
        c_inst = QColor("cyan") if self.dark_mode else QColor("blue")
        c_label = QColor("#f1fa8c") if self.dark_mode else QColor("#795e26")
        c_dir = QColor("gray")
        c_com = QColor("lightgreen") if self.dark_mode else QColor("green")
        fmt_reg = QTextCharFormat(); fmt_reg.setForeground(c_reg); fmt_reg.setFontWeight(QFont.Bold)
        fmt_inst = QTextCharFormat(); fmt_inst.setForeground(c_inst)
        fmt_label = QTextCharFormat(); fmt_label.setForeground(c_label); fmt_label.setFontWeight(QFont.Bold)
        fmt_dir = QTextCharFormat(); fmt_dir.setForeground(c_dir)
        fmt_com = QTextCharFormat(); fmt_com.setForeground(c_com)
        self.rules.append((re.compile(r'%[a-z0-9]+'), fmt_reg))
        insts = ["mov", "push", "pop", "call", "ret", "add", "sub", "jmp", "je", "jne", "cmp", "lea", "nop", "xor", "inc", "dec"]
        for word in insts: self.rules.append((re.compile(r'\b' + word + r'[a-z]*\b'), fmt_inst))
        self.rules.append((re.compile(r'^\s*\.?[a-zA-Z0-9_]+:'), fmt_label))
        self.rules.append((re.compile(r'\.[a-z_]+'), fmt_dir))
        self.rules.append((re.compile(r'[#;].*$'), fmt_com))

class LineNumberArea(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.codeEditor = editor
    def sizeHint(self): return QSize(self.codeEditor.lineNumberAreaWidth(), 0)
    def paintEvent(self, event): self.codeEditor.lineNumberAreaPaintEvent(event)

class CodeEditor(QPlainTextEdit):
    current_line_changed = pyqtSignal(int)

    def __init__(self, parent=None, mode="c", font=None):
        super().__init__(parent)
        self.current_file = None 
        self.mode = mode
        self.dark_mode = False
        self.line_mapping = None 
        
        if mode == "asm":
            self.highlighter = AssemblyHighlighter(self.document())
        else:
            self.highlighter = CppSyntaxHighlighter(self.document())
        self.highlighter.update_rules()
        
        if font: self.setFont(font)
        else: self.setFont(QFont("Courier", 11))
        
        self.lineNumberArea = LineNumberArea(self)
        self.blockCountChanged.connect(self.updateLineNumberAreaWidth)
        self.updateRequest.connect(self.updateLineNumberArea)
        self.cursorPositionChanged.connect(self.highlightCurrentLine)
        self.updateLineNumberAreaWidth(0)
        self.highlightCurrentLine()


    def clear_heatmap(self):
        """ Clears any profiling colors """
        self.setExtraSelections([])

    def apply_heatmap(self, line_counts):
        """ 
        Colors lines based on execution count.
        line_counts: dict { line_number (int): execution_count (int) }
        """
        if not line_counts: return
        
        # Use log scale so a loop running 1000 times doesn't wash out a line running 10 times
        max_count = max(line_counts.values())
        if max_count == 0: return
        log_max = math.log(max_count + 1)

        selections = []
        for line_num, count in line_counts.items():
            if count == 0: continue # Don't highlight unexecuted code
            
            # Calculate intensity (0.0 to 1.0)
            intensity = math.log(count + 1) / log_max
            
            # Create color: Red = Hot (1.0), Green = Cold (0.0)
            # We use an alpha channel (transparency) so text remains readable
            if self.dark_mode:
                # Dark mode: Dark Red (hot) to Dark Green (cold)
                r = int(255 * intensity)
                g = int(255 * (1 - intensity))
                color = QColor(r, g, 0, 100) 
            else:
                # Light mode: Light Red (hot) to Light Green (cold)
                r = int(255 * intensity)
                g = int(255 * (1 - intensity))
                color = QColor(r, g, 0, 80)

            sel = QTextEdit.ExtraSelection()
            sel.format.setBackground(color)
            sel.format.setProperty(QTextFormat.FullWidthSelection, True)
            
            # Select the line
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.Start)
            cursor.movePosition(QTextCursor.Down, QTextCursor.MoveAnchor, line_num - 1)
            sel.cursor = cursor
            selections.append(sel)
            
        self.setExtraSelections(selections)

    def set_theme(self, is_dark):
        self.dark_mode = is_dark
        self.highlighter.set_theme(is_dark)
        self.highlightCurrentLine()
        self.update()

    def lineNumberAreaWidth(self):
        digits = 1
        max_num = max(1, self.blockCount())
        while max_num >= 10: max_num //= 10; digits += 1
        space = 3 + self.fontMetrics().width('9') * digits
        return space

    def updateLineNumberAreaWidth(self, _):
        self.setViewportMargins(self.lineNumberAreaWidth(), 0, 0, 0)

    def updateLineNumberArea(self, rect, dy):
        if dy: self.lineNumberArea.scroll(0, dy)
        else: self.lineNumberArea.update(0, rect.y(), self.lineNumberArea.width(), rect.height())
        if rect.contains(self.viewport().rect()): self.updateLineNumberAreaWidth(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.lineNumberArea.setGeometry(QRect(cr.left(), cr.top(), self.lineNumberAreaWidth(), cr.height()))

    def lineNumberAreaPaintEvent(self, event):
        painter = QPainter(self.lineNumberArea)
        bg = QColor(40, 40, 40) if self.dark_mode else Qt.lightGray
        fg = Qt.white if self.dark_mode else Qt.black
        painter.fillRect(event.rect(), bg)
        block = self.firstVisibleBlock()
        blockNumber = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(blockNumber + 1)
                painter.setPen(fg)
                painter.drawText(0, top, self.lineNumberArea.width(), self.fontMetrics().height(), Qt.AlignRight, number)
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            blockNumber += 1

    def highlightCurrentLine(self):
        extraSelections = []
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            if self.dark_mode: lineColor = QColor(60, 60, 60)
            else: lineColor = QColor(Qt.yellow).lighter(180)
            selection.format.setBackground(lineColor)
            selection.format.setProperty(QTextFormat.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            extraSelections.append(selection)
        
        self.setExtraSelections(extraSelections)
        line = self.textCursor().blockNumber() + 1
        self.current_line_changed.emit(line)

    def highlight_mapped_lines(self, lines_ranges):
        current_selections = self.extraSelections()
        new_selections = [s for s in current_selections if s.format.property(QTextFormat.FullWidthSelection)]
        map_color = QColor(100, 100, 255, 80) if self.dark_mode else QColor(200, 200, 255, 150)
        for start_idx, end_idx in lines_ranges:
            sel = QTextEdit.ExtraSelection()
            sel.format.setBackground(map_color)
            sel.format.setProperty(QTextFormat.FullWidthSelection, True)
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.Start)
            cursor.movePosition(QTextCursor.NextBlock, QTextCursor.MoveAnchor, start_idx)
            for _ in range(end_idx - start_idx + 1):
                cursor.movePosition(QTextCursor.NextBlock, QTextCursor.KeepAnchor)
            sel.cursor = cursor
            new_selections.append(sel)
        self.setExtraSelections(new_selections)

class InteractiveGraphView(QGraphicsView):
    def __init__(self, svg_path, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.svg_item = QGraphicsSvgItem(svg_path)
        self.scene.addItem(self.svg_item)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        factor = 1.15 if zoom_in else 1 / 1.15
        self.scale(factor, factor)

class TabbedCFGViewer(QWidget):
    def __init__(self, per_func_cfgs, temp_path, parent=None, is_dom_tree=False):
        super().__init__(parent)
        self.temp_path = temp_path
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        for func_name, cfg_text in per_func_cfgs.items():
            if is_dom_tree:
                graph = parse_cfg_to_graph(cfg_text)
                if not graph.nodes: continue
                entry = list(graph.nodes)[0]
                if '2' in graph.nodes: entry = '2'
                doms = compute_dominators(graph, entry)
                dot = generate_dom_tree_dot(doms, graph.labels)
            else:
                dot = parse_cfg_to_dot(cfg_text)
            tab = self.create_graph_tab(dot)
            self.tabs.addTab(tab, func_name)

    def create_graph_tab(self, dot_source):
        try:
            fd, dot_path = tempfile.mkstemp(dir=self.temp_path, suffix=".dot", text=True)
            with os.fdopen(fd, 'w') as f: f.write(dot_source)
            svg_path = dot_path + ".svg"
            subprocess.run(["dot", "-Tsvg", dot_path, "-o", svg_path], check=True)
            view = InteractiveGraphView(svg_path)
            return view
        except Exception as e:
            err_widget = QTextEdit()
            err_widget.setHtml(f"<h3>Error rendering:</h3><pre>{e}</pre>")
            return err_widget

class GimpleDiffViewer(QWidget):
    def __init__(self, diff, parent=None):
        super().__init__(parent)
        self.sections = self.segment_diff(diff)
        self.sidebar = QListWidget()
        self.sidebar.addItems(self.sections.keys())
        self.sidebar.currentTextChanged.connect(self.show_section)
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.text_view.setFont(QFont("Courier", 10))
        splitter = QSplitter()
        splitter.addWidget(self.sidebar); splitter.addWidget(self.text_view)
        splitter.setStretchFactor(1, 2)
        layout = QVBoxLayout(); layout.setContentsMargins(0,0,0,0); layout.addWidget(splitter)
        self.setLayout(layout)
        self.sidebar.setCurrentRow(0)
    def segment_diff(self, diff):
        sections = {"All": diff}
        keyword_map = { "Dead Code": ["eliminate", "unused", "dce"], "Inlining": ["inline"], "Constant Folding": ["fold", "constant"], "Loop": ["loop", "unroll"], "Strength": ["strength"], "Reordering": ["reorder"] }
        for line in diff:
            matched = False
            for sec, kws in keyword_map.items():
                if any(k in line.lower() for k in kws): sections.setdefault(sec, []).append(line); matched=True
            if not matched: sections.setdefault("Misc", []).append(line)
        return sections
    def show_section(self, section_name): self.text_view.setHtml(self.format_diff_lines(self.sections.get(section_name, [])))
    def format_diff_lines(self, lines):
        html = []
        for line in lines:
            line = line.replace("<", "&lt;").replace(">", "&gt;")
            if line.startswith("+") and not line.startswith("+++"): html.append(f'<span style="color:green;">{line}</span>')
            elif line.startswith("-") and not line.startswith("---"): html.append(f'<span style="color:red;">{line}</span>')
            elif line.startswith("@@"): html.append(f'<span style="color:blue;">{line}</span>')
            else: html.append(f'<span>{line}</span>')
        return "<pre>" + "\n".join(html) + "</pre>"

class RtlDiffViewer(GimpleDiffViewer): 
    def segment_rtl(self, diff): return self.segment_diff(diff) 

class OptimizationTimelineViewer(QWidget):
    def __init__(self, pass_diffs, mode="gimple", parent=None):
        super().__init__(parent)
        self.pass_diffs = pass_diffs
        self.mode = mode
        layout = QVBoxLayout()
        splitter = QSplitter()
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(250)
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget); self.right_layout.setContentsMargins(0,0,0,0)
        for (n1, n2), _ in pass_diffs: self.sidebar.addItem(f"{n1} → {n2}")
        self.sidebar.currentRowChanged.connect(self.display_diff)
        splitter.addWidget(self.sidebar); splitter.addWidget(self.right_widget); splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter); self.setLayout(layout)
        if self.sidebar.count() > 0: self.sidebar.setCurrentRow(0)
    def display_diff(self, index):
        _, diff = self.pass_diffs[index]
        for i in reversed(range(self.right_layout.count())): 
            w = self.right_layout.itemAt(i).widget(); 
            if w: w.setParent(None)
        self.right_layout.addWidget(GimpleDiffViewer(diff))

# ========================================================
# 4. MAIN WINDOW
# ========================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name
        self.setWindowTitle("Compiler Explorer - IDE Edition")
        self.resize(1400, 900)
        
        self.compiler_path = "gcc"
        self.cpp_compiler_path = "g++"
        self.compiler_type = "GCC" 
        self.editor_font = QFont("Courier", 11)
        self.is_dark_mode = False

        # --- Tool Bar ---
        toolbar = QToolBar("Build Settings")
        self.addToolBar(toolbar)
        
        toolbar.addWidget(QLabel(" Standard: "))
        self.std_combo = QComboBox()
        self.std_combo.addItems(["Default", "c99", "c11", "c17", "c2x", "c++11", "c++14", "c++17", "c++20", "c++23"])
        toolbar.addWidget(self.std_combo)
        
        toolbar.addWidget(QLabel("  Flags: "))
        self.flags_input = QLineEdit()
        self.flags_input.setPlaceholderText("-fno-inline -march=native ...")
        toolbar.addWidget(self.flags_input)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.main_splitter)

        self.tabs_left = QTabWidget()
        self.tabs_left.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tabs_left.customContextMenuRequested.connect(lambda p: self.show_tab_context_menu(p, self.tabs_left))
        self.tabs_left.setTabsClosable(True)
        self.tabs_left.tabCloseRequested.connect(lambda i: self.close_tab(i, self.tabs_left))
        self.main_splitter.addWidget(self.tabs_left)

        self.tabs_right = QTabWidget()
        self.tabs_right.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tabs_right.customContextMenuRequested.connect(lambda p: self.show_tab_context_menu(p, self.tabs_right))
        self.tabs_right.setTabsClosable(True)
        self.tabs_right.tabCloseRequested.connect(lambda i: self.close_tab(i, self.tabs_right))
        self.main_splitter.addWidget(self.tabs_right)
        self.tabs_right.hide()
        
        self.sync_scroll_enabled = False
        self.find_dialog = None
        self.optimization_level = "-O2"
        self.create_menu()
        self.add_editor_tab("Untitled.c", "", target_tabs=self.tabs_left)

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Open File", self.open_file, "Ctrl+O")
        file_menu.addAction("Save Current File", self.save_current_file, "Ctrl+S")
        file_menu.addSeparator()
        file_menu.addAction("New Tab", lambda: self.add_editor_tab("Untitled.c", "", self.tabs_left), "Ctrl+N")
        
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction("Format Code (clang-format)", self.format_code, "Ctrl+I")
        edit_menu.addAction("Find and Replace", self.open_find_replace, "Ctrl+F")
        edit_menu.addAction("Preferences", self.open_preferences)

        view_menu = menubar.addMenu("View")
        view_menu.addAction("Move Tab to Right Split", self.move_current_tab_right)
        view_menu.addAction("Move Tab to Left Split", self.move_current_tab_left)
        self.sync_action = QAction("Synchronize Scrolling", self, checkable=True)
        self.sync_action.triggered.connect(self.toggle_sync_scroll)
        view_menu.addAction(self.sync_action)
        view_menu.addAction("Close Right Split", self.close_right_split)
        self.theme_action = QAction("Toggle Dark Mode", self, checkable=True)
        self.theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(self.theme_action)

        build_menu = menubar.addMenu("Build & Analyze")
        build_menu.addAction("Compile Only", self.build_only, "Ctrl+B")
        build_menu.addAction("Run Static Analysis (cppcheck)", self.run_cppcheck)
        build_menu.addAction("Run with Sanitizers", self.run_sanitizers)
        build_menu.addAction("Analyze Binary Size", self.run_size_analysis)
        build_menu.addAction("Generate Assembly (.s)", self.build_assembly)
        build_menu.addAction("Generate CFG", self.build_with_cfg, "Ctrl+Shift+B")
        build_menu.addAction("Generate Dominator Tree", self.build_dom_tree)
        build_menu.addAction("View AST", self.build_ast)
        build_menu.addAction("Generate Header Dependency Graph", self.view_header_graph)
        build_menu.addAction("Run Profiling (Heatmap)", self.run_profiling)

        opt_menu = menubar.addMenu("Optimization")
        self.opt_group = {}
        def set_opt_level(opt):
            def setter():
                self.optimization_level = opt
                for level, action in self.opt_group.items(): action.setChecked(level == opt)
            return setter
        for level in ["-O0", "-O1", "-O2", "-O3", "-Og", "-Os", "-Ofast"]:
            action = QAction(level, self, checkable=True)
            if level == self.optimization_level: action.setChecked(True)
            action.triggered.connect(set_opt_level(level))
            opt_menu.addAction(action)
            self.opt_group[level] = action

        analyze_menu = build_menu.addMenu("Optimization Dumps")
        analyze_menu.addAction("Analyze GIMPLE Passes", lambda: self.build_and_show_optimizations("gimple"))
        analyze_menu.addAction("Analyze RTL Passes", lambda: self.build_and_show_optimizations("rtl"))


    def view_header_graph(self):
        p, b, v = self._prepare_compile()
        if not v: return
        cmd_bin = self.get_compiler_for_file(p)
        flags = self.get_build_flags()
        
        # -H prints includes to stderr. -fsyntax-only prevents full compile (faster)
        cmd = [cmd_bin, '-H', '-fsyntax-only', p] + flags
        
        try:
            # We capture Stderr because that's where -H writes
            res = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse
            graph = parse_include_tree(res.stderr)
            if not graph.nodes:
                QMessageBox.information(self, "Info", "No headers found or parsing failed.")
                return

            # Convert to DOT format
            dot_lines = ["digraph Includes {", "rankdir=LR;", "node [shape=box, style=filled, fillcolor=\"#e1f5fe\"];"]
            for n, label in graph.labels.items():
                dot_lines.append(f'"{n}" [label="{label}"];')
            for src, targets in graph.succs.items():
                for tgt in targets:
                    dot_lines.append(f'"{src}" -> "{tgt}";')
            dot_lines.append("}")
            dot_source = "\n".join(dot_lines)
            
            # Reuse your existing TabbedCFGViewer to show it
            # We wrap it in a dummy dict to fit the viewer's API
            dummy_container = {"Include Hierarchy": ""} # The viewer usually parses text, but we will inject the DOT manually
            
            # Hack: Instantiate viewer but override the tab creation
            viewer = TabbedCFGViewer({}, self.temp_path)
            view_widget = viewer.create_graph_tab(dot_source)
            viewer.tabs.addTab(view_widget, "Include Graph")
            
            self.show_in_right_pane(viewer, f"Includes: {b}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ----------------------------------------
    # FEATURE: Profiling (Hotspots)
    # ----------------------------------------
    def run_profiling(self):
        # 1. Prepare: Get paths
        p, b, v = self._prepare_compile()
        if not v: return
        
        base_name = os.path.splitext(b)[0]
        temp_src = os.path.join(self.temp_path, b)
        
        try:
            shutil.copy(p, temp_src)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not copy source to temp: {e}")
            return

        cmd_bin = self.get_compiler_for_file(p)
        exe = os.path.join(self.temp_path, base_name)
        if sys.platform == "win32": exe += ".exe"
        
        # Clean previous gcov files
        for f in glob.glob(os.path.join(self.temp_path, "*.gc*")):
            try: os.remove(f)
            except: pass
            
        # 2. Compile LOCALLY
        flags = [self.optimization_level, '-fprofile-arcs', '-ftest-coverage'] + self.get_build_flags()
        
        try:
            subprocess.run([cmd_bin, b] + flags + ['-o', exe], cwd=self.temp_path, check=True, capture_output=True)
            
            # 3. Run the code
            # CHANGE: check=False. We expect this might crash (SIGFPE/SIGSEGV).
            # We want to proceed to gcov analysis even if it dies.
            proc = subprocess.run([exe], cwd=self.temp_path, check=False, capture_output=True)
            
            if proc.returncode != 0:
                # Log the crash but don't stop.
                # Note: On Unix, negative return codes indicate termination by signal (e.g., -8 is SIGFPE)
                print(f"Binary crashed with code {proc.returncode}. Attempting to recover coverage data...")

            # 4. Run gcov
            subprocess.run(['gcov', b], cwd=self.temp_path, check=True, capture_output=True)
            
            # 5. Parse .gcov file
            gcov_file = os.path.join(self.temp_path, f"{b}.gcov")
            
            if not os.path.exists(gcov_file):
                # Fallback search
                candidates = glob.glob(os.path.join(self.temp_path, "*.gcov"))
                if candidates:
                    gcov_file = candidates[0]
                else:
                    # If the program crashed too early, .gcda might not be written, so gcov fails.
                    QMessageBox.warning(self, "Incomplete Data", 
                        f"Program crashed (Code {proc.returncode}) and no coverage data was flushed.\n"
                        "Note: Crashes often prevent GCOV from saving data.")
                    return
                
            line_data = {}
            with open(gcov_file, 'r') as f:
                for line in f:
                    parts = line.split(':', 2)
                    if len(parts) < 3: continue
                    
                    count_str = parts[0].strip()
                    line_num_str = parts[1].strip()
                    
                    if count_str == '-' or count_str == '#####': continue 
                    
                    try:
                        count = int(count_str)
                        line_num = int(line_num_str)
                        line_data[line_num] = count
                    except ValueError: pass

            editor = self.get_current_editor()
            editor.clear_heatmap()
            editor.apply_heatmap(line_data)
            
            msg = f"Profiling complete for {b}."
            if proc.returncode != 0:
                msg += f"\n(Warning: Program crashed with code {proc.returncode})"
            QMessageBox.information(self, "Profiling", msg)
            
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode() if e.stderr else str(e)
            QMessageBox.critical(self, "Profiling Failed", f"Step failed: {e.cmd}\n\nError:\n{err_msg}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # --- Code Formatting ---
    def format_code(self):
        editor = self.get_current_editor()
        if not editor: return
        
        # Check for clang-format
        if shutil.which("clang-format") is None:
            QMessageBox.warning(self, "Error", "clang-format not found in PATH.")
            return
            
        code = editor.toPlainText()
        try:
            # Run clang-format via stdin/stdout
            p = subprocess.Popen(
                ["clang-format", "-style=Google"], 
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True
            )
            out, err = p.communicate(input=code)
            
            if p.returncode == 0:
                # Preserve scroll/cursor if possible
                cursor = editor.textCursor()
                sb = editor.verticalScrollBar().value()
                editor.setPlainText(out)
                editor.verticalScrollBar().setValue(sb)
                editor.setTextCursor(cursor)
            else:
                QMessageBox.critical(self, "Formatting Failed", err)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # --- Assembly to Source Mapping ---
    def on_source_line_changed(self, line_num):
        sender_widget = self.sender()
        if self.tabs_left.indexOf(sender_widget) != -1: target_tabs = self.tabs_right
        else: target_tabs = self.tabs_left
        
        if target_tabs.isVisible():
            widget = target_tabs.currentWidget()
            if isinstance(widget, CodeEditor) and widget.mode == "asm" and widget.line_mapping:
                ranges = widget.line_mapping.get(line_num, [])
                widget.highlight_mapped_lines(ranges)

    # --- Navigation ---
    def navigate_to_code(self, filename, line):
        target_tabs = self.tabs_left 
        found_widget = None
        def search_tabs(tabs):
            for i in range(tabs.count()):
                w = tabs.widget(i)
                if isinstance(w, CodeEditor) and w.current_file:
                    if os.path.basename(w.current_file) == os.path.basename(filename): return w
            return None
        found_widget = search_tabs(self.tabs_left)
        if not found_widget:
            found_widget = search_tabs(self.tabs_right)
            if found_widget: target_tabs = self.tabs_right
        if not found_widget:
            current = self.get_current_editor()
            if current and current.current_file and os.path.basename(current.current_file) == filename:
                found_widget = current
        
        if found_widget:
            if self.tabs_left.indexOf(found_widget) != -1: self.tabs_left.setCurrentWidget(found_widget)
            else: self.tabs_right.setCurrentWidget(found_widget)
            cursor = found_widget.textCursor()
            cursor.movePosition(QTextCursor.Start)
            cursor.movePosition(QTextCursor.Down, QTextCursor.MoveAnchor, line - 1)
            found_widget.setTextCursor(cursor)
            found_widget.highlightCurrentLine()
            found_widget.setFocus()
        else: QMessageBox.information(self, "Navigation", f"Could not find open editor for {filename}")

    # --- Binary Size Analysis ---
    def run_size_analysis(self):
        p, b, v = self._prepare_compile()
        if not v: return
        cmd_bin = self.get_compiler_for_file(p)
        obj_path = os.path.join(self.temp_path, f"{b}.o")
        flags = self.get_build_flags()
        
        try:
            subprocess.run([cmd_bin, '-c', p, self.optimization_level] + flags + ['-o', obj_path], check=True)
            res = subprocess.run(['nm', '-S', '--size-sort', '-t', 'd', obj_path], capture_output=True, text=True)
            if res.returncode != 0: QMessageBox.critical(self, "Analysis Failed", f"nm error: {res.stderr}"); return
            html = """<html><head><style>body { font-family: sans-serif; padding: 20px; background-color: #f0f0f0; }
            table { border-collapse: collapse; width: 100%; background: white; }
            th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .bar-container { width: 100%; background-color: #e0e0e0; height: 20px; border-radius: 4px; }
            .bar { height: 100%; background-color: #4CAF50; text-align: right; padding-right: 5px; color: white; border-radius: 4px; }
            </style></head><body><h2>Binary Size Analysis</h2><table><tr><th>Symbol</th><th>Type</th><th>Size (Bytes)</th><th>Visualization</th></tr>"""
            max_size = 1
            rows = []
            for line in res.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 4:
                    size = int(parts[1])
                    rows.append({'size': size, 'type': parts[2], 'name': parts[3]})
                    if size > max_size: max_size = size
            for r in reversed(rows):
                width = (r['size'] / max_size) * 100
                html += f"<tr><td>{r['name']}</td><td>{r['type']}</td><td>{r['size']}</td><td><div class='bar-container'><div class='bar' style='width:{width}%'></div></div></td></tr>"
            html += "</table></body></html>"
            view = QWebEngineView()
            view.setHtml(html)
            self.show_in_right_pane(view, f"Size: {b}")
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    # --- Build Helpers ---
    def show_in_right_pane(self, widget, title):
        self.tabs_right.show()
        if self.main_splitter.sizes()[1] == 0:
             w = self.main_splitter.width()
             self.main_splitter.setSizes([w // 2, w // 2])
        self.tabs_right.addTab(widget, title)
        self.tabs_right.setCurrentWidget(widget)

    def get_compiler_for_file(self, filepath):
        if filepath.endswith(('.cpp', '.cc', '.cxx', '.hpp')):
            return self.cpp_compiler_path
        return self.compiler_path

    def get_build_flags(self):
        flags = self.flags_input.text().split()
        std = self.std_combo.currentText()
        if std != "Default":
            flags.append(f"-std={std}")
        return flags

    def _prepare_compile(self):
        editor = self.get_current_editor()
        if not editor or not editor.current_file:
            QMessageBox.warning(self, "Warning", "Save file first.")
            self.save_as_file()
            editor = self.get_current_editor()
            if not editor or not editor.current_file: return None, None, False
        with open(editor.current_file, 'w') as f: f.write(editor.toPlainText())
        return editor.current_file, os.path.basename(editor.current_file), True

    # --- Build Methods ---
    def build_only(self):
        p, _, v = self._prepare_compile()
        if not v: return
        cmd_bin = self.get_compiler_for_file(p)
        flags = self.get_build_flags()
        try:
            r = subprocess.run([cmd_bin, p, self.optimization_level] + flags + ['-o', os.path.join(self.temp_path, 'a.out')], capture_output=True, text=True)
            if r.returncode==0: QMessageBox.information(self, "Success", "Compiled!")
            else: QMessageBox.critical(self, "Failed", r.stderr)
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def build_assembly(self):
        p, b, v = self._prepare_compile()
        if not v: return
        cmd_bin = self.get_compiler_for_file(p)
        out = os.path.join(self.temp_path, f"{b}.s")
        flags = self.get_build_flags()
        try:
            subprocess.run([cmd_bin, '-S', '-fverbose-asm', '-g', p, self.optimization_level] + flags + ['-o', out], check=True)
            with open(out, 'r') as f: c = f.read()
            mapping = parse_assembly_mapping(c)
            stats = analyze_asm_stats(c) # Get stats
            
            self.tabs_right.show()
            if self.main_splitter.sizes()[1] == 0: self.main_splitter.setSizes([self.main_splitter.width()//2]*2)
            
            # Add editor with stats header
            full_content = stats + c
            self.add_editor_tab(f"{b} (ASM)", full_content, self.tabs_right, mode="asm")
            asm_editor = self.tabs_right.currentWidget()
            asm_editor.line_mapping = mapping
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def build_ast(self):
        p, b, v = self._prepare_compile()
        if not v: return
        cmd_bin = self.get_compiler_for_file(p)
        flags = self.get_build_flags()
        
        if self.compiler_type == "Clang":
            cmd = [cmd_bin, '-Xclang', '-ast-dump', '-fsyntax-only', p] + flags
            try:
                r = subprocess.run(cmd, capture_output=True, text=True)
                self.show_in_right_pane(CodeEditor(mode="c", font=self.editor_font), f"AST (Clang): {b}")
                self.tabs_right.currentWidget().setPlainText(r.stdout)
            except Exception as e: QMessageBox.critical(self, "Error", str(e))
        else:
            try:
                subprocess.run([cmd_bin, '-fdump-tree-original', p] + flags, cwd=self.temp_path, check=True)
                f_list = glob.glob(os.path.join(self.temp_path, "*original*"))
                if f_list:
                    with open(f_list[0], 'r') as fl: c = fl.read()
                    self.show_in_right_pane(CodeEditor(mode="c", font=self.editor_font), f"AST (GCC): {b}")
                    self.tabs_right.currentWidget().setPlainText(c)
                else: QMessageBox.warning(self, "Error", "AST dump not found (check if compiler supports it).")
            except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def build_with_cfg(self):
        if self.compiler_type == "Clang":
            QMessageBox.warning(self, "Warning", "CFG visualization is GCC-only currently.")
            return
        p, b, v = self._prepare_compile()
        if not v: return
        cmd_bin = self.get_compiler_for_file(p)
        flags = self.get_build_flags()
        try:
            subprocess.run([cmd_bin, p, self.optimization_level, '-fdump-tree-cfg'] + flags, cwd=self.temp_path, check=True)
            f_list = glob.glob(os.path.join(self.temp_path, "*cfg*"))
            if f_list:
                with open(f_list[0], 'r') as fl: c = fl.read()
                self.show_in_right_pane(TabbedCFGViewer(extract_cfgs_per_function(c), self.temp_path), f"CFG: {b}")
            else: QMessageBox.warning(self, "Error", "CFG dump not found.")
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def run_sanitizers(self):
        p, b, v = self._prepare_compile()
        if not v: return
        cmd_bin = self.get_compiler_for_file(p)
        exe = os.path.join(self.temp_path, 'sanitized_app')
        flags = self.get_build_flags()
        try:
            r = subprocess.run([cmd_bin, p, self.optimization_level, '-fsanitize=address,undefined', '-g'] + flags + ['-o', exe], capture_output=True, text=True)
            if r.returncode != 0: QMessageBox.critical(self, "Build Failed", r.stderr); return
            r2 = subprocess.run([exe], capture_output=True, text=True)
            log_view = LogViewer(font=self.editor_font)
            log_view.setPlainText(f"STDOUT:\n{r2.stdout}\n\nSTDERR:\n{r2.stderr}")
            log_view.navigation_requested.connect(self.navigate_to_code)
            self.show_in_right_pane(log_view, f"Run: {b}")
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def run_cppcheck(self):
        filepath, base, valid = self._prepare_compile()
        if not valid: return
        cmd = ["cppcheck", "--enable=all", "--force", filepath]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            log_view = LogViewer(font=self.editor_font)
            log_view.setPlainText(res.stderr)
            log_view.navigation_requested.connect(self.navigate_to_code)
            self.show_in_right_pane(log_view, f"Cppcheck: {base}")
        except FileNotFoundError: QMessageBox.critical(self, "Error", "cppcheck not found. Please install it.")

    def build_dom_tree(self):
        if self.compiler_type == "Clang":
            QMessageBox.warning(self, "Warning", "Dominator Tree is GCC-only currently.")
            return
        p, b, v = self._prepare_compile()
        if not v: return
        cmd_bin = self.get_compiler_for_file(p)
        flags = self.get_build_flags()
        try:
            for f in glob.glob(os.path.join(self.temp_path, "*")): 
                try: os.remove(f); 
                except: pass
            subprocess.run([cmd_bin, p, self.optimization_level, '-fdump-tree-cfg'] + flags, cwd=self.temp_path, check=True)
            files = glob.glob(os.path.join(self.temp_path, "*cfg*"))
            if not files: return
            with open(files[0], 'r') as f: cfg_text = f.read()
            self.show_in_right_pane(TabbedCFGViewer(extract_cfgs_per_function(cfg_text), self.temp_path, is_dom_tree=True), f"Dominator Tree: {b}")
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def build_and_show_optimizations(self, mode):
        if self.compiler_type == "Clang":
            QMessageBox.warning(self, "Warning", "Optimization Dumps are GCC-only currently.")
            return
        p, b, v = self._prepare_compile()
        if not v: return
        cmd_bin = self.get_compiler_for_file(p)
        flag = "-fdump-tree-all" if mode=="gimple" else "-fdump-rtl-all"
        pat = "*t.*" if mode=="gimple" else "*r.*"
        flags = self.get_build_flags()
        try:
            for f in glob.glob(os.path.join(self.temp_path, "*")): 
                try: os.remove(f); 
                except: pass
            subprocess.run([cmd_bin, p, self.optimization_level, flag] + flags, cwd=self.temp_path, check=True)
            files = sorted(glob.glob(os.path.join(self.temp_path, f"*{pat}")), key=lambda x: int(re.search(r'\.(\d+)[tr]\.', x).group(1)) if re.search(r'\.(\d+)[tr]\.', x) else 999)
            if len(files) < 2: return
            passes = [(-1, f.split('.')[-1], f) for f in files]
            self.show_in_right_pane(OptimizationTimelineViewer(generate_pass_diffs(passes), mode=mode), f"{mode.upper()}: {b}")
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    # --- Tab Management ---
    def show_tab_context_menu(self, point, tab_widget):
        index = tab_widget.tabBar().tabAt(point)
        if index >= 0:
            menu = QMenu(self)
            if tab_widget == self.tabs_left:
                menu.addAction("Move to Right Split").triggered.connect(lambda: self.move_tab_to_split(index, self.tabs_left, self.tabs_right))
            else:
                menu.addAction("Move to Left Split").triggered.connect(lambda: self.move_tab_to_split(index, self.tabs_right, self.tabs_left))
            menu.addAction("Close Tab").triggered.connect(lambda: self.close_tab(index, tab_widget))
            menu.exec_(tab_widget.mapToGlobal(point))

    def move_tab_to_split(self, index, from_tabs, to_tabs):
        widget = from_tabs.widget(index)
        title = from_tabs.tabText(index)
        from_tabs.removeTab(index)
        to_tabs.addTab(widget, title)
        to_tabs.setCurrentWidget(widget)
        to_tabs.show()
        if to_tabs == self.tabs_right and self.main_splitter.sizes()[1] == 0: self.main_splitter.setSizes([self.main_splitter.width()//2]*2)
        if from_tabs == self.tabs_right and from_tabs.count() == 0: from_tabs.hide()

    def move_current_tab_right(self):
        idx = self.tabs_left.currentIndex()
        if idx >= 0: self.move_tab_to_split(idx, self.tabs_left, self.tabs_right)
    def move_current_tab_left(self):
        idx = self.tabs_right.currentIndex()
        if idx >= 0: self.move_tab_to_split(idx, self.tabs_right, self.tabs_left)
    def close_right_split(self):
        while self.tabs_right.count(): 
            w=self.tabs_right.widget(0); t=self.tabs_right.tabText(0); self.tabs_right.removeTab(0); self.tabs_left.addTab(w, t)
        self.tabs_right.hide()

    def add_editor_tab(self, title, content, target_tabs=None, filepath=None, mode="c"):
        if not target_tabs: target_tabs = self.tabs_left
        ed = CodeEditor(mode=mode, font=self.editor_font)
        ed.set_theme(self.is_dark_mode)
        ed.setPlainText(content); ed.current_file = filepath
        
        if mode == "c":
            ed.current_line_changed.connect(self.on_source_line_changed)
            
        target_tabs.setCurrentIndex(target_tabs.addTab(ed, title))

    # --- Boilerplate (Open, Save, Theme, Sync, etc. - kept compact) ---
    def toggle_theme(self, checked):
        self.is_dark_mode = checked
        if checked: self.apply_dark_theme()
        else: self.apply_light_theme()
        for t in [self.tabs_left, self.tabs_right]:
            for i in range(t.count()):
                if isinstance(t.widget(i), CodeEditor): t.widget(i).set_theme(checked)
    def apply_dark_theme(self):
        p = QPalette()
        p.setColor(QPalette.Window, QColor(53, 53, 53)); p.setColor(QPalette.WindowText, Qt.white)
        p.setColor(QPalette.Base, QColor(25, 25, 25)); p.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        p.setColor(QPalette.ToolTipBase, Qt.white); p.setColor(QPalette.ToolTipText, Qt.white)
        p.setColor(QPalette.Text, Qt.white); p.setColor(QPalette.Button, QColor(53, 53, 53))
        p.setColor(QPalette.ButtonText, Qt.white); p.setColor(QPalette.BrightText, Qt.red)
        p.setColor(QPalette.Link, QColor(42, 130, 218)); p.setColor(QPalette.Highlight, QColor(42, 130, 218))
        p.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.instance().setPalette(p)
    def apply_light_theme(self): QApplication.instance().setPalette(QApplication.style().standardPalette())
    def open_preferences(self): PreferencesDialog(self).exec_()
    def open_find_replace(self): 
        if self.find_dialog: self.find_dialog.close()
        self.find_dialog = FindReplaceDialog(self.get_current_editor(), self); self.find_dialog.show()
    def update_editor_font(self, font):
        self.editor_font = font
        for t in [self.tabs_left, self.tabs_right]:
            for i in range(t.count()): 
                if isinstance(t.widget(i), CodeEditor): t.widget(i).setFont(font)
    def toggle_sync_scroll(self, checked):
        self.sync_scroll_enabled = checked
        if checked: 
            l, r = self.tabs_left.currentWidget(), self.tabs_right.currentWidget()
            if isinstance(l, CodeEditor) and isinstance(r, CodeEditor):
                l.verticalScrollBar().valueChanged.connect(self._sync_l2r)
                r.verticalScrollBar().valueChanged.connect(self._sync_r2l)
        else:
            try: self.tabs_left.currentWidget().verticalScrollBar().disconnect(); self.tabs_right.currentWidget().verticalScrollBar().disconnect()
            except: pass
    def _sync_l2r(self, v): 
        r = self.tabs_right.currentWidget()
        if r and self.sync_scroll_enabled: r.verticalScrollBar().setValue(int(v/self.tabs_left.currentWidget().verticalScrollBar().maximum()*r.verticalScrollBar().maximum()))
    def _sync_r2l(self, v):
        l = self.tabs_left.currentWidget()
        if l and self.sync_scroll_enabled: l.verticalScrollBar().setValue(int(v/self.tabs_right.currentWidget().verticalScrollBar().maximum()*l.verticalScrollBar().maximum()))
    def close_tab(self, idx, tabs):
        if tabs.count()>1: tabs.removeTab(idx)
        elif tabs==self.tabs_right: tabs.removeTab(0); tabs.hide()
        else: w=tabs.widget(0); w.clear(); w.current_file=None; tabs.setTabText(0, "Untitled.c")
    def get_current_editor(self):
        w = self.tabs_left.currentWidget()
        return w if isinstance(w, CodeEditor) else self.tabs_right.currentWidget() if isinstance(self.tabs_right.currentWidget(), CodeEditor) else None
    def open_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open File", "", "C/C++ Files (*.c *.cpp *.cc *.cxx *.h *.hpp)")
        if p:
            with open(p, 'r') as f: c = f.read()
            cur = self.get_current_editor()
            if cur and not cur.current_file and not cur.toPlainText().strip() and self.tabs_left.indexOf(cur)!=-1:
                cur.setPlainText(c); cur.current_file=p; self.tabs_left.setTabText(self.tabs_left.currentIndex(), os.path.basename(p))
            else: self.add_editor_tab(os.path.basename(p), c, self.tabs_left, p)
    def save_current_file(self):
        ed = self.get_current_editor()
        if ed and ed.current_file:
            with open(ed.current_file, 'w') as f: f.write(ed.toPlainText())
            QMessageBox.information(self, "Saved", f"Saved {os.path.basename(ed.current_file)}")
        else: self.save_as_file()
    def save_as_file(self):
        ed = self.get_current_editor()
        if not ed: return
        p, _ = QFileDialog.getSaveFileName(self, "Save As", "", "C/C++ Files (*.c *.cpp *.cc *.cxx *.h *.hpp)")
        if p:
            with open(p, 'w') as f: f.write(ed.toPlainText())
            ed.current_file=p; idx=self.tabs_left.indexOf(ed)
            if idx!=-1: self.tabs_left.setTabText(idx, os.path.basename(p))
            else: self.tabs_right.setTabText(self.tabs_right.indexOf(ed), os.path.basename(p))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())