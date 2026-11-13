"""Tiny pop-up to retype an episode prompt. Prints the chosen prompt on stdout.

    python scripts/prompt_dialog.py "<current prompt>" <goal colour> <goal shape>

Why a separate process? Tk and the Genesis viewer both talk to the X server, and using both from
one process (on different threads) crashes Xlib. A separate process has its own X connection.
(Why plain widgets instead of tkinter.simpledialog? Its modal "grab" crashes the Tk build that
ships with uv's Python under XWayland.)
"""

import sys
import tkinter as tk


def main():
    current, colour, shape = sys.argv[1], sys.argv[2], sys.argv[3]
    result = {"text": current}
    root = tk.Tk()
    root.title("Edit prompt")
    root.attributes("-topmost", True)
    tk.Label(root, text=f"The goal is the {colour} {shape}. Type the instruction you want to use:",
             padx=10, pady=6).pack(anchor="w")
    var = tk.StringVar(value=current)
    entry = tk.Entry(root, textvariable=var, width=50, font=("TkDefaultFont", 12))
    entry.pack(padx=10, fill="x")
    warn = tk.Label(root, text="", fg="#b45309", padx=10, justify="left")
    warn.pack(anchor="w")
    state = {"warned_for": None}

    def ok(_event=None):
        text = " ".join(var.get().strip().split())
        if not text:
            return
        if colour not in text.lower() and state["warned_for"] != text:
            # Same shapes come in several colours, so a prompt without the colour may be ambiguous.
            warn.config(text=f'This prompt does not mention "{colour}" - the model may not know which object '
                             "you mean.\nPress OK again to use it anyway.")
            state["warned_for"] = text
            return
        result["text"] = text
        root.destroy()

    buttons = tk.Frame(root, pady=8)
    buttons.pack()
    tk.Button(buttons, text="OK", width=10, command=ok).pack(side="left", padx=4)
    tk.Button(buttons, text="Cancel", width=10, command=root.destroy).pack(side="left", padx=4)
    root.bind("<Return>", ok)
    root.bind("<Escape>", lambda _e: root.destroy())
    entry.focus_force()
    entry.select_range(0, "end")
    root.mainloop()
    print(result["text"])


if __name__ == "__main__":
    main()
