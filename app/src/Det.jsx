import React, { useState } from "react";
import axios from "axios";
import './Det.css';

function Det() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append("file", file);

    // Make the POST request to Flask server
    axios
      .post("http://localhost:5000/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((response) => {
        // On successful response, set the data
        console.log("Data received:", response.data);
        setInfo(response.data);  // Store the received data
      })
      .catch((err) => {
        // Handle errors
        console.error("Error uploading file:", err);
        setError("Failed to upload or process the file.");
      });
  };

  return (
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
          <h2 className="text">Upload your driver's license!</h2>
          <input type="file"  accept=".jpg, .jpeg, .png" required />
        </div>
        <button className="submit-button" onClick={handleFileChange}>Submit</button>
        {error && <p className="error">{error}</p>}

        {info && (
          <div>
            <h3>Extracted Data:</h3>
            <table>
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(info.data).map(([key, value]) => (
                  <tr key={key}>
                    <td>{key}</td>
                    <td>{JSON.stringify(value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}

export default Det;
