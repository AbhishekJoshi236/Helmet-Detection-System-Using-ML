const express = require('express');
const { spawn } = require('child_process');
const multer = require('multer');
const path = require('path');

const app = express();
const port = 3000;

// Serve static files from the current directory
app.use(express.static(__dirname));

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/')
    },
    filename: function (req, file, cb) {
        cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname))
    }
});

const upload = multer({ storage: storage });

app.post('/startDetection', upload.single('video'), (req, res) => {
    const videoPath = req.file.path;
    const pythonProcess = spawn('python', ['both.py', videoPath]);

    pythonProcess.stdout.on('data', (data) => {
        console.log(`stdout: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
    });

    res.sendFile(path.join(__dirname, 'pages', 'detection_started.html'));
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
