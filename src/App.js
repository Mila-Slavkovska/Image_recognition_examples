import React, { useRef, useState, useEffect } from 'react';
import * as tf from "@tensorflow/tfjs";
import * as handpose from "@tensorflow-models/handpose";
import Webcam from 'react-webcam';
import './App.css';
import { drawHand } from './utils/utilities';
import { NeuralNetwork } from 'brain.js';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const [trainingData, setTrainingData] = useState([]); // Store training data
  const [trained, setTrained] = useState(false); // Flag to check if model is trained
  const [status, setStatus] = useState(''); // For status updates
  const [currentGesture, setCurrentGesture] = useState(''); // Current gesture being recorded
  const [dataCollectionActive, setDataCollectionActive] = useState(false); // To control data collection

  // Initialize neural network
  const network = new NeuralNetwork({
    hiddenLayers: [128], // Adjust this for complexity
  });

  // Convert landmarks to input array
  const convertLandmarksToInput = (landmarks) => {
    return landmarks.flat();
  };

  // Run handpose detection
  const runHandpose = async () => {
    const net = await handpose.load();
    setInterval(() => {
      detectHand(net);
    }, 100);
  };

  // Gather hand landmarks for a specific gesture
  const gatherDataForClass = async (gestureName, duration = 500) => {
    setStatus(`Gathering data for ${gestureName}...`);
    
    if (webcamRef.current && webcamRef.current.video.readyState === 4) {
      console.log('Webcam is ready. Capturing landmarks...');
    
      const video = webcamRef.current.video;
      const net = await handpose.load();
      
      const startTime = Date.now();
      
      const captureLandmarks = async () => {
        if (Date.now() - startTime < duration) {
          const hand = await net.estimateHands(video);
      
          if (hand.length > 0) {
            const landmarks = hand[0].landmarks.flat();
            console.log('Captured landmarks:', landmarks);
      
            setTrainingData(prevData => [
              ...prevData,
              { input: convertLandmarksToInput(landmarks), output: { [gestureName]: 1 } }
            ]);
      
            setStatus(`Data for ${gestureName} gathered!`);
          } else {
            setStatus(`No hand detected for ${gestureName}.`);
          }
          
          setTimeout(captureLandmarks, 500); // Continue capturing every 500ms
        } else {
          setStatus(`Data collection for ${gestureName} stopped.`);
          console.log(`Data collection for ${gestureName} stopped.`);
        }
      };
      
      captureLandmarks();
    } else {
      setStatus(`Webcam not ready for ${gestureName}.`);
      console.log(`Webcam not ready for ${gestureName}.`);
    }
  };

  // Detect hand landmarks and make predictions
  const detectHand = async (net) => {
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
  
      const hand = await net.estimateHands(video);
  
      if (hand.length > 0) {
        const landmarks = hand[0].landmarks.flat();
  
        if (dataCollectionActive && currentGesture) {
          setTrainingData(prevData => {
            const newData = [
              ...prevData,
              { input: convertLandmarksToInput(landmarks), output: { [currentGesture]: 1 } }
            ];
            console.log('Training data collected:', newData);
            return newData;
          });
          setStatus(`Data for ${currentGesture} gathered!`);
        }
  
        if (trained) {
          try {
            const input = convertLandmarksToInput(landmarks);
            const output = network.run(input);
            console.log('Model output:', output);

            // Log the predicted gesture to the console
            if (output.А > 0.5) {
              console.log('Predicted gesture: А');
            } else if (output.Б > 0.5) {
              console.log('Predicted gesture: Б');
            } else if (output.В > 0.5) {
              console.log('Predicted gesture: В');
            } else if (output.Г > 0.5) {
              console.log('Predicted gesture: Г');
            } else if (output.Д > 0.5) {
              console.log('Predicted gesture: Д');
            } else {
              console.log('No gesture recognized');
            }
          } catch (error) {
            console.error('Prediction error:', error);
          }
        }
  
        const ctx = canvasRef.current.getContext("2d");
        drawHand(hand, ctx);
      } else {
        //console.log('No hand detected');
      }
    } else {
      console.log('Webcam not ready or unavailable');
    }
  };

  // Start prediction loop
  const startPrediction = async () => {
    if (webcamRef.current && webcamRef.current.video.readyState === 4) {
      console.log('Webcam is ready for prediction.');
      
      const video = webcamRef.current.video;
      const net = await handpose.load();
      
      const predictGesture = async () => {
        const hand = await net.estimateHands(video);
        
        if (hand.length > 0) {
          const landmarks = hand[0].landmarks.flat();
          console.log('Landmarks for prediction:', landmarks);
          
          const prediction = network.run(convertLandmarksToInput(landmarks));
          console.log('Prediction:', prediction);
          
          // Log the predicted gesture to the console
          if (prediction.А > 0.5) {
            console.log('Predicted gesture: А');
          } else if (prediction.Б > 0.5) {
            console.log('Predicted gesture: Б');
          } else if (prediction.В > 0.5) {
            console.log('Predicted gesture: В');
          } else if (prediction.Г > 0.5) {
            console.log('Predicted gesture: Г');
          } else if (prediction.Д > 0.5) {
            console.log('Predicted gesture: Д');
          } else {
            console.log('No gesture recognized');
          }
        } else {
          console.log('No hand detected.');
        }
        
        // Continue predicting every 500ms
        setTimeout(predictGesture, 500);
      };
      
      predictGesture();
    } else {
      console.log('Webcam not ready for prediction.');
    }
  };

  // Train the neural network with gathered data
  const handleTrainModel = () => {
    if (trainingData.length > 0) {
      console.log('Training data:', trainingData);
      try {
        network.train(trainingData, {
          log: (details) => {
            console.log('Training log:', details); // Should log training progress
          },
          iterations: 100,
          logPeriod: 10,
        });
        setTrained(true);
        setStatus("Model trained successfully!");
        startPrediction(); // Start prediction after training
      } catch (error) {
        console.error('Training error:', error);
      }
    } else {
      setStatus("No data to train on!");
    }
  };

  const checkWebcamReady = () => {
    if (webcamRef.current && webcamRef.current.video) {
      const video = webcamRef.current.video;
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        console.log('Webcam is ready');
        return true;
      } else {
        console.log('Webcam is not ready');
        return false;
      }
    } else {
      console.log('Webcam reference is not available');
      return false;
    }
  };

  useEffect(() => { runHandpose(); }, []);
  useEffect(() => {
    console.log('Training data:', trainingData);
  }, [trainingData]);
  useEffect(() => {
    const interval = setInterval(() => {
      if (checkWebcamReady()) {
        clearInterval(interval);
        runHandpose();
      }
    }, 1000); // Check every second
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <Webcam
          ref={webcamRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 640,
            height: 480
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
            zindex: 9,
            width: 640,
            height: 480
          }}
        />

        {/* Buttons for gathering data and training model */}
        <div id="buttons">
          <button onClick={() => gatherDataForClass("А")}>Gather "А" Data</button>
          <button onClick={() => gatherDataForClass("Б")}>Gather "Б" Data</button>
          <button onClick={() => gatherDataForClass("В")}>Gather "В" Data</button>
          <button onClick={() => gatherDataForClass("Г")}>Gather "Г" Data</button>
          <button onClick={() => gatherDataForClass("Д")}>Gather "Д" Data</button>
          <button onClick={handleTrainModel}>Train Model</button>
        </div>

        {/* Status updates */}
        <p>{status}</p>
      </header>
    </div>
  );
}

export default App;
