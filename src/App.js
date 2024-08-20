import React, {useRef, useState, useEffect} from 'react';
import * as tf from "@tensorflow/tfjs";
import * as handpose from "@tensorflow-models/handpose";
import Webcam from 'react-webcam';
import './App.css';
import { drawHand } from './utils/utilities';

import * as fp from "fingerpose";
import victory from "./victory.png";
import thumbs_up from "./thumbs_up.png";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const [emoji, setEmoji] = useState(null);
  const images = {thumbs_up:thumbs_up, victory:victory};
  //console.log("in app empji ", emoji)

  useEffect(() => {
    console.log("Updated emoji state:", emoji);
  }, [emoji]);

  const runHandpose = async () => {
    const net = await handpose.load();

    setInterval(() => {
      detectHand(net)
    }, 100);
  }

  const detectHand = async (net) => {
      if(typeof webcamRef.current !== "undefined" && webcamRef.current !== null &&
        webcamRef.current.video.readyState === 4){
        const video = webcamRef.current.video;
        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;

        webcamRef.current.video.width = videoWidth;
        webcamRef.current.video.height = videoHeight;

        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;

        const hand = await net.estimateHands(video);
        // console.log(hand);

        if(hand.length > 0){
          const GE = new fp.GestureEstimator([
            fp.Gestures.VictoryGesture,
            fp.Gestures.ThumbsUpGesture,
          ]);

          const gesture = await GE.estimate(hand[0].landmarks, 8);
          //console.log(gesture);
          if (gesture.gestures !== undefined && gesture.gestures.length > 0){
            const confidence = gesture.gestures.map(
              (prediction) => prediction.score
            );
  
            //console.log(confidence)
            //console.log(gesture)
  
            const maxConfidence = confidence.indexOf(
              Math.max.apply(null, confidence)
            );
            //console.log(maxConfidence)
  
            const detectedGesture = gesture.gestures[maxConfidence].name;
            setEmoji(detectedGesture);
            console.log("detected gesture", detectedGesture);
            //console.log(gesture.gestures[maxConfidence].name);
            //console.log(emoji);
          }
          else{
            setEmoji(null);
          }
        }
        else{
          setEmoji(null);
        }
        const ctx = canvasRef.current.getContext("2d");
        drawHand(hand, ctx);
      }
  }

  useEffect(()=>{runHandpose()},[]);

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
        {emoji !== null ? (
          <img
            src={images[emoji]}
            style={{
              position: "absolute",
              marginLeft: "auto",
              marginRight: "auto",
              left: 400,
              bottom: 500,
              right: 0,
              textAlign: "center",
              height: 100,
            }}
          />
        ) : (
          ""
        )}
      </header>
    </div>
  );
}

export default App;
