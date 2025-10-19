
import * as THREE from "../libs/three.js/build/three.module.js"
import JSON5 from "../libs/json5-2.1.3/json5.mjs";
const { ipcRenderer, dialog } = require('electron');



ipcRenderer.on('open-metadata', (_evt, filepath) => {
  console.log("Loading metadata.json:", filepath);
  loadDroppedPointcloud(filepath);   // you already have this function
});



export function loadDroppedPointcloud(cloudjsPath){
	const folderName = cloudjsPath.replace(/\\/g, "/").split("/").reverse()[1];

	Potree.loadPointCloud(cloudjsPath).then(e => {
		let pointcloud = e.pointcloud;
		let material = pointcloud.material;

		pointcloud.name = folderName;

		viewer.scene.addPointCloud(pointcloud);

		let hasRGBA = pointcloud.getAttributes().attributes.find(a => a.name === "rgba") !== undefined
		if(hasRGBA){
			pointcloud.material.activeAttributeName = "rgba";
		}else{
			pointcloud.material.activeAttributeName = "color";
		}

		material.size = 1;
		material.pointSizeType = Potree.PointSizeType.ADAPTIVE;

		viewer.zoomTo(e.pointcloud);
	});
}
