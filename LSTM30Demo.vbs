Option Explicit
Dim shell, files, root, pythonw
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
root = files.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
pythonw = root & "\.venv\Scripts\pythonw.exe"
If Not files.FileExists(pythonw) Then
    MsgBox "Python environment is missing. Install the GUI and ML dependencies first.", 48, "DockDack LSTM30"
    WScript.Quit 1
End If
' Opening the GUI never silently enables orders. Use its explicit ON confirmation.
shell.Run Chr(34) & pythonw & Chr(34) & " -m examples.run_lstm30_gui --top-us100 --buy-threshold 0.4 --close-all-before-minutes 5 --confirm-close-all DEMO_CLOSE_ALL_SELLABLE --quantity 1 --max-krw 500000 --max-usd 1000 --domestic-checkpoint models/lstm30/domestic.pt --us-checkpoint models/lstm30/us.pt", 1, False
