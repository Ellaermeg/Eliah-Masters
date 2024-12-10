#Requires AutoHotkey v2.0  ; Requires AutoHotkey v2.0 or higher

F1::
{
    MouseGetPos(&xPos, &yPos)  ; Get the current mouse position
    MsgBox("Mouse Coordinates:`nX: " xPos "`nY: " yPos)  ; Show the coordinates in a message box
}
