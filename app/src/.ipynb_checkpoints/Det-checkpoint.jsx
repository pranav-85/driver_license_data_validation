import React from "react";
import { useState } from "react";
import './Det.css'
function Det(){

    const[info, setInfo] = useState();

    
    return(
        <>
            <header>
                <div className="logo">
                <span>A</span>
                <span>p</span>
                <span>p</span>
                </div>

            </header>
            
            <div className="container">
                <div className="input-class">
                    <h2 class="text">Upload your driver's license!</h2>

                    <input type="file" id="file-input" accept=".jpg, .jpeg, .png, .pdf"/>
                    
                </div>
            </div>
        </>
    );
}

export default Det