import React, { useRef, useState, useEffect, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import "./App.css";
import { drawRect } from "./utilities";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [facingMode, setFacingMode] = useState("user"); // Front-facing by default

  // Main function
  const runCoco = useCallback(async () => {
    const net = await tf.loadGraphModel(
      "https://tesnsorflowjsrealtimemodel.s3.us-east.cloud-object-storage.appdomain.cloud/model.json"
    );

    setInterval(() => {
      detect(net);
    }, 16.7);
  }, []);

  const detect = async (net) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;

      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      const img = tf.browser.fromPixels(video);
      const resized = tf.image.resizeBilinear(img, [640, 480]);
      const casted = resized.cast("int32");
      const expanded = casted.expandDims(0);
      const obj = await net.executeAsync(expanded);

      const boxes = await obj[1].array();
      const classes = await obj[2].array();
      const scores = await obj[4].array();

      const ctx = canvasRef.current.getContext("2d");

      requestAnimationFrame(() => {
        drawRect(boxes[0], classes[0], scores[0], 0.8, videoWidth, videoHeight, ctx);
      });

      tf.dispose(img);
      tf.dispose(resized);
      tf.dispose(casted);
      tf.dispose(expanded);
      tf.dispose(obj);
    }
  };

  useEffect(() => {
    runCoco();
  }, [runCoco]);

  return (
    <div className="App">
      <header className="App-header">
        {/* Toggle Front/Back Camera */}
        <button
          onClick={() =>
            setFacingMode((prevMode) => (prevMode === "user" ? "environment" : "user"))
          }
          style={{
            position: "absolute",
            top: 10,
            zIndex: 10,
            padding: "10px 20px",
            fontSize: "16px",
          }}
        >
          Switch Camera
        </button>

        <Webcam
          ref={webcamRef}
          muted={true}
          videoConstraints={{ facingMode }}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 9,
            width: 640,
            height: 480,
          }}
        />

        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 8,
            width: 640,
            height: 480,
          }}
        />
      </header>
    </div>
  );
}

export default App;
