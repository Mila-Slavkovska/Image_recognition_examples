const fingerJoints = {
    thumb: [0, 1, 2, 3, 4],
    indexFinger: [0, 5, 6, 7, 8],
    middleFinger: [0, 9, 10, 11, 12],
    ringFinger: [0, 13, 14, 15, 16],
    pinky: [0, 17, 18, 19, 20],
};

export const drawLandmarks = (landmarksArray, myCanvas) => {
    const canvas = myCanvas;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    landmarksArray.forEach((landmarks) => {
    //   console.log("lanmarks array: ", landmarksArray)
       for(let i=0; i<Object.keys(fingerJoints).length; i++){
              let finger = Object.keys(fingerJoints)[i];

              for(let j=0; j<fingerJoints[finger].length-1; j++){
                  const firstJointIndex = fingerJoints[finger][j];
                  const secondJointIndex = fingerJoints[finger][j+1];

                  ctx.beginPath();
                  ctx.moveTo(
                      landmarks[firstJointIndex].x * canvas.width,
                      landmarks[firstJointIndex].y * canvas.height
                  );
                  ctx.lineTo(
                      landmarks[secondJointIndex].x * canvas.width,
                      landmarks[secondJointIndex].y * canvas.height
                  );

                  ctx.strokeStyle = "plum";
                  ctx.lineWidth = 3;
                  ctx.stroke();
              }
          }

      landmarks.forEach((landmark) => {        
        const x = landmark.x * canvas.width;
        const y = landmark.y * canvas.height;

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2*Math.PI); 
        ctx.fillStyle = "indigo";
        ctx.fill();
      });
    });
  };