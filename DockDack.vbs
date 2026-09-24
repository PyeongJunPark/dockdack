Option Explicit
Dim shell, files, root, python, pythonw, result, base
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
root = files.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
base = root & "\.venv-ml-cuda\Scripts\"
If Not files.FileExists(base & "pythonw.exe") Then
    base = files.GetParentFolderName(root) & "\dockdack\.venv-ml-cuda\Scripts\"
End If
If Not files.FileExists(base & "pythonw.exe") Then
    base = root & "\.venv\Scripts\"
End If
python = base & "python.exe"
pythonw = base & "pythonw.exe"
If Not files.FileExists(pythonw) Then
    MsgBox "Run 'uv sync --extra gui --extra prototype' in the project folder first.", 48, "DockDack setup"
    WScript.Quit 1
End If
result = shell.Run(Chr(34) & python & Chr(34) & " -c " & Chr(34) & "from examples.run_desktop_gui import configure_local_dependencies; configure_local_dependencies(); import dockdack.v00_app" & Chr(34), 0, True)
If result <> 0 Then
    MsgBox "GUI dependencies are missing. Run 'uv sync --extra gui --extra prototype' in the project folder.", 48, "DockDack setup"
    WScript.Quit 1
End If
shell.Run Chr(34) & pythonw & Chr(34) & " -m examples.run_desktop_gui --no-model --external-model mark1-prototype --external-model mark1-1-prototype", 0, False
