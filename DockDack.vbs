Option Explicit
Dim shell, files, root, python, pythonw, result
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
root = files.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
python = root & "\.venv\Scripts\python.exe"
pythonw = root & "\.venv\Scripts\pythonw.exe"
If Not files.FileExists(pythonw) Then
    MsgBox "Run 'uv sync --extra gui' in the project folder first.", 48, "DockDack setup"
    WScript.Quit 1
End If
result = shell.Run(Chr(34) & python & Chr(34) & " -c " & Chr(34) & "import dockdack.gui" & Chr(34), 0, True)
If result <> 0 Then
    MsgBox "GUI dependencies are missing. Run 'uv sync --extra gui' in the project folder.", 48, "DockDack setup"
    WScript.Quit 1
End If
shell.Run Chr(34) & pythonw & Chr(34) & " -m dockdack.v00_app", 1, False
