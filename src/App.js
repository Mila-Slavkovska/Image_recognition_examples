import React, { useRef, useEffect, useState } from "react";
import * as tf from '@tensorflow/tfjs';
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import hand_landmarker_task from "./models/hand_landmarker.task";
import { drawLandmarks } from "./utils/utilities";
import "./App.css";

const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let mobilenet = undefined;
let model = undefined;
let gatherDataState = STOP_DATA_GATHER;

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [handPresence, setHandPresence] = useState(null);
  const [loaded, setLoaded] = useState(false);
  const [CLASS_NAMES, setClassNames] = useState([]);
  const [status, setStatus] = useState("");
  const [predictionText, setPredictionText] = useState("");
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

    const classes = Array.from(uniqueClasses);
    
    model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [1024], units: 128, activation: 'relu'}));
    model.add(tf.layers.dense({units: classes.length, activation: 'softmax'}));

    model.summary();

    model.compile({
      optimizer: 'adam',
      loss: (classes.length === 2) ? 'binaryCrossentropy': 'categoricalCrossentropy', 
      metrics: ['accuracy']  
    });
    
  }, []); 
  
  async function handleTrainAndPredict() {
    predict = false;
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
    let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
    let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
    let inputsAsTensor = tf.stack(trainingDataInputs);
    
    let results = await model.fit(inputsAsTensor, oneHotOutputs, {shuffle: true, batchSize: 5, epochs: 10, 
        callbacks: {onEpochEnd: logProgress} });
    
    outputsAsTensor.dispose();
    oneHotOutputs.dispose();
    inputsAsTensor.dispose();
    predict = true;
    predictLoop();
  }

  function logProgress(epoch, logs) {
    console.log('Data for epoch ' + epoch, logs);
  }  
  
  function predictLoop() {
    if (predict) {
      tf.tidy(function() {
        let videoFrameAsTensor = tf.browser.fromPixels(videoRef.current).div(255);
        let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor,[MOBILE_NET_INPUT_HEIGHT, 
            MOBILE_NET_INPUT_WIDTH], true);
          
        let imageFeatures = mobilenet.predict(resizedTensorFrame.expandDims());
        let prediction = model.predict(imageFeatures).squeeze();
        let highestIndex = prediction.argMax().arraySync();
        let predictionArray = prediction.arraySync();
  
        setPredictionText('Prediction: ' + CLASS_NAMES[highestIndex] + ' with ' + Math.floor(predictionArray[highestIndex] * 100) + '% confidence');
      });
  
      window.requestAnimationFrame(predictLoop);
    }
  }  
  
  function handleReset() {
    predict = false;
    examplesCount.length = 0;
    for (let i = 0; i < trainingDataInputs.length; i++) {
      trainingDataInputs[i].dispose();
    }
    trainingDataInputs.length = 0;
    trainingDataOutputs.length = 0;
    setStatus('No data collected');
    
    console.log('Tensors in memory: ' + tf.memory().numTensors);
  }

  function gatherDataForClass(e) {
    let classNumber = +e.target.getAttribute("data-1hot");

    if (e.type === "mousedown") {
      gatherDataState = classNumber;
      dataGatherLoop();
    } else if (e.type === "mouseup") {
      gatherDataState = STOP_DATA_GATHER;
    }

    dataGatherLoop();
  }  

  function dataGatherLoop(){
    if (gatherDataState !== STOP_DATA_GATHER && videoRef.current && videoRef.current.readyState >= 2){
      let imageFeatures = tf.tidy(function(){
        let videoFrameAsTensor = tf.browser.fromPixels(videoRef.current);
        let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT, 
            MOBILE_NET_INPUT_WIDTH], true);
        let normalizedTensorFrame = resizedTensorFrame.div(255);
        return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
      });

      trainingDataInputs.push(imageFeatures);
      trainingDataOutputs.push(gatherDataState);  

      if (examplesCount[gatherDataState] === undefined) {
        examplesCount[gatherDataState] = 0;
      }
      examplesCount[gatherDataState]++;

      let newStatus = '';
      for (let n = 0; n < CLASS_NAMES.length; n++) {
        newStatus += CLASS_NAMES[n] + ' data count: ' + examplesCount[n] + '. ';
      }
      setStatus(newStatus);
      
      window.requestAnimationFrame(dataGatherLoop);
    } 
  }
  
  return (
    <>
      <h1>{loaded? "MobileNet v3 loaded successfully!" : "Loading..."}</h1>
      <h1>Is there a Hand? {handPresence ? "Yes" : "No"}</h1>
      <p>{predictionText}</p>
      <div style={{ position: "relative" }}>
      <p style={{position: "absolute", left: 2}}>{status.split(". ").map(s => <li key={s}>{s}</li>)}</p>
        <video
          id="video"
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
          <button id="train" onClick={handleTrainAndPredict}>Train &amp; Predict!</button>
          <button id="reset" onClick={handleReset}>Reset</button> <br/>
          <button 
          className="dataCollector" data-1hot="0" data-name="Class A"
          onMouseUp={gatherDataForClass} 
          onMouseDown={gatherDataForClass}
          >
            Gather Class A Data
          </button>

          <button 
          className="dataCollector" data-1hot="1" data-name="Class B"
          onMouseUp={gatherDataForClass} 
          onMouseDown={gatherDataForClass}
          >
            Gather Class B Data
          </button>

          <button className="dataCollector" data-1hot="2" data-name="Class V"
          onMouseUp={gatherDataForClass} 
          onMouseDown={gatherDataForClass}
          >
            Gather Class V Data
          </button>
        </div>
    </>
  );
}

export default App;
