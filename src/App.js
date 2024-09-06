import React, { useRef, useEffect, useState } from "react";
import * as tf from '@tensorflow/tfjs';
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import hand_landmarker_task from "./models/hand_landmarker.task";
import { drawLandmarks } from "./utils/utilities";
import "./App.css";

const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [handPresence, setHandPresence] = useState(null);
  const [loaded, setLoaded] = useState(false);
  const [CLASS_NAMES, setClassNames] = useState([]);

  let mobilenet = undefined; //MAY HAVE TO MOVE
  let gatherDataState = STOP_DATA_GATHER;
  let videoPlaying = false;
  let trainingDataInputs = [];
  let trainingDataOutputs = [];
  let examplesCount = [];
  let predict = false;


  useEffect(() => {
    let handLandmarker;
    let animationFrameId;

    const initializeHandDetection = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: hand_landmarker_task },
          numHands: 2,
          runningMode: "video",
        });
        detectHands();
      } catch (error) {
        console.error("Error initializing hand detection:", error);
      }
    };

    const detectHands = () => {
      if (videoRef.current && videoRef.current.readyState >= 2) {
        const detections = handLandmarker.detectForVideo(
          videoRef.current,
          performance.now()
        );
        setHandPresence(detections.handednesses.length > 0);

        if (detections.landmarks) {
          drawLandmarks(detections.landmarks, canvasRef.current);
        }
      }
      requestAnimationFrame(detectHands);
    };

    const startWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        videoRef.current.srcObject = stream;
        await initializeHandDetection();
      } catch (error) {
        console.error("Error accessing webcam:", error);
      }
    };

    startWebcam();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      }
      if (handLandmarker) {
        handLandmarker.close();
      }
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, []);

  useEffect(() => {
    const buttons = document.querySelectorAll('button[data-name]');
    const uniqueClasses = new Set();
    
    buttons.forEach(button => {
      uniqueClasses.add(button.getAttribute('data-name'));
    });
    
    setClassNames(Array.from(uniqueClasses));

  
    async function loadMobileNetFeatureModel() {
      const URL = 
        'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
      
      mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});
      setLoaded(true);
      
      tf.tidy(function () {
        let answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
        console.log(answer.shape);
      });
    }

    loadMobileNetFeatureModel();

  }, []);

  function handleEnableCam() {
    // TODO: Fill this out later in the codelab!
    console.log("web cam enableing")
  }
  
  
  function handleTrainAndPredict() {
    // TODO: Fill this out later in the codelab!
    console.log("train and predict")
  }
  
  
  function handleReset() {
    // TODO: Fill this out later in the codelab!
    console.log("reset")
  }

  function gatherDataForClass(e) {
    // TODO: Fill this out later in the codelab!
    console.log(e.target)
  }  
  

  return (
    <>
      <h1>{loaded? "MobileNet v3 loaded successfully!" : "Loading..."}</h1>
      <h1>Is there a Hand? {handPresence ? "Yes" : "No"}</h1>
      <div style={{ position: "relative" }}>
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 640,
            height: 480,
          }}
        ></video>
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 640,
            height: 480,
          }}
        ></canvas>
      </div>
      <div id="buttons">
          <button id="enableCam" onClick={handleEnableCam}>Enable Webcam</button>
          <button id="train" onClick={handleTrainAndPredict}>Train &amp; Predict!</button>
          <button id="reset" onClick={handleReset}>Reset</button> <br/>
          <button 
          className="dataCollector" data-1hot="0" data-name="Class A"
          onMouseUp={gatherDataForClass} onMouseDown={gatherDataForClass}
          >
            Gather Class A Data
          </button>

          <button 
          className="dataCollector" data-1hot="1" data-name="Class B"
          onMouseUp={gatherDataForClass} onMouseDown={gatherDataForClass}
          >
            Gather Class B Data
          </button>

          <button className="dataCollector" data-1hot="2" data-name="Class V"
          onMouseUp={gatherDataForClass} onMouseDown={gatherDataForClass}
          >
            Gather Class V Data
          </button>
        </div>
    </>
  );
}

export default App;
