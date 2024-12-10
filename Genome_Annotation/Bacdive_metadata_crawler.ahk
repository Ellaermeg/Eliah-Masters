#Requires AutoHotkey v2.0  ; Requires AutoHotkey v2.0 or higher


F1::{
    
    Loop 3
        {
            ; Coordinates of the button
            x := 1427
            y := 475
            Click, %x%, %y%  ; Click at the specified position
            Sleep, 1000  ; Wait for 2 seconds before clicking again

            x := 1603
            y := 349
            Click, %x%, %y% 
            Sleep, 1000

            x := 2584
            y := 337
            Click, %x%, %y%
            Sleep, 5000
        }

}
    