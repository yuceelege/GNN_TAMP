
torso: {X: "t(0 -2.6 0.25)", shape: ssBox, size: [0.5, .3, 0.5, .1],contact:0}
r_endeffector: {shape: sphere, size: [.05, .05, .05, .06], color: [0, 0, 0], contact:1}


(torso r_endeffector): { joint: free, pre: "t(0 0 0.6) d(45 0 0 1)", post: "t(0 0 .1)", Q: "T d(0 0 0 0)"}
