const electron = require('electron')
const app = electron.app
const BrowserWindow = electron.BrowserWindow
const Menu = electron.Menu;
const MenuItem = electron.MenuItem;
const remote = electron.remote;
const { ipcMain, dialog } = require('electron');

const path = require('path')
const url = require('url')

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow

function createWindow () {
	// Create the browser window.
	mainWindow = new BrowserWindow({
		width: 1920,
		height: 1080,
		webPreferences: {
			nodeIntegration: true,
			backgroundThrottling: false,
			contextIsolation: false,
			enableRemoteModule: true,
			webSecurity: false,
		}
 
	})

	// and load the index.html of the app.
	// mainWindow.loadURL(url.format({
	// 	pathname: path.join(__dirname, 'index.html'),
	// 	protocol: 'file:',
	// 	slashes: true
	// }));
	mainWindow.loadFile(path.join(__dirname, 'index.html'));
	// mainWindow.webContents.openDevTools();

	// mainWindow.webContents.once('did-finish-load', async () => {
	// 	const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
	// 		title: 'Open Potree project',
	// 		buttonLabel: 'Open',
	// 		properties: ['openFile'],
	// 		filters: [
	// 		{ name: 'Potree projects', extensions: ['json'] }, // metadata.json
	// 		{ name: 'All files', extensions: ['*'] }
	// 		]
	// 	});

	// 	if (!canceled && filePaths.length) {
	// 		mainWindow.webContents.send('open-metadata', filePaths[0]);
	// 	}
	// });



	//let menu = new Menu();
	//let menuItemWindow = new MenuItem({label: "Window"});
	//let menuItemToggleDevTools = new MenuItem({label: 'Toggle Developer Tools', click() { 
	//		//remote.getCurrentWindow().toggleDevTools();
	//		mainWindow.webContents.toggleDevTools();
	//	}});
	////menuItemWindow.append(menuItemToggleDevTools);
	//menuItemWindow.submenu = [menuItemToggleDevTools];

	//menu.append(menuItemWindow);

	let template = [
		{
			label: "Window",
			submenu: [
				{label: "Reload", click() { mainWindow.webContents.reloadIgnoringCache() }},
				{label: "Toggle Developer Tools", click() { mainWindow.webContents.toggleDevTools() }},
			]
		}
	];

	

	let menu = Menu.buildFromTemplate(template);
	mainWindow.setMenu(menu);


	{
		const { ipcMain } = require('electron');

		ipcMain.on('asynchronous-message', (event, arg) => {
			console.log(arg) // prints "ping"
			event.reply('asynchronous-reply', 'pong')
		})

		ipcMain.on('synchronous-message', (event, arg) => {
			console.log(arg) // prints "ping"
			event.returnValue = 'pong'
		})
	}

	ipcMain.handle('pick-pcd', async () => {
		return await dialog.showOpenDialog({
			title: "Select a PCD file",
			properties: ['openFile'],
			filters: [{ name: 'Point Clouds', extensions: ['pcd'] }]
		});
	});


	// 	process.stdout.on('data', (data) => {
	// 		console.log(`stdout: ${data}`);
	// 	});
	// }
	


	// require('remote').getCurrentWindow().toggleDevTools();


	// Open the DevTools.
	//mainWindow.webContents.openDevTools();

	// Emitted when the window is closed.
	mainWindow.on('closed', function () {
		// Dereference the window object, usually you would store windows
		// in an array if your app supports multi windows, this is the time
		// when you should delete the corresponding element.
		mainWindow = null
	})
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', function () {
	// On OS X it is common for applications and their menu bar
	// to stay active until the user quits explicitly with Cmd + Q
	if (process.platform !== 'darwin') {
		app.quit()
	}
})

app.on('activate', function () {
	// On OS X it's common to re-create a window in the app when the
	// dock icon is clicked and there are no other windows open.
	if (mainWindow === null) {
		createWindow()
	}
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
