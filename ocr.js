var ocrDemo = {
    CANVAS_WIDTH: 200,
    PIXEL_WIDTH: 10,
    get TRANSLATED_WIDTH() {
        return this.CANVAS_WIDTH / this.PIXEL_WIDTH;
    },
    data: [],
    trainArray: [],
    trainingRequestCount: 0,
    BATCH_SIZE: 10,
    HOST: "http://localhost",
    PORT: "8080",
    BLUE: '#0000FF', // Define the color used for drawing the grid

    drawGrid: function(ctx) {
        for (var x = this.PIXEL_WIDTH; x < this.CANVAS_WIDTH; x += this.PIXEL_WIDTH) {
            ctx.strokeStyle = this.BLUE;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.CANVAS_WIDTH);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, x); // Draw horizontal lines
            ctx.lineTo(this.CANVAS_WIDTH, x);
            ctx.stroke();
        }
    },

    onMouseMove: function(e, ctx, canvas) {
        if (!canvas.isDrawing) {
            return;
        }
        this.fillSquare(ctx, 
            e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    },

    onMouseDown: function(e, ctx, canvas) {
        canvas.isDrawing = true;
        this.fillSquare(ctx, 
            e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    },

    onMouseUp: function(canvas) { // Pass the canvas as an argument
        canvas.isDrawing = false;
    },

    fillSquare: function(ctx, x, y) {
        var xPixel = Math.floor(x / this.PIXEL_WIDTH);
        var yPixel = Math.floor(y / this.PIXEL_WIDTH);
        this.data[(xPixel * this.TRANSLATED_WIDTH + yPixel)] = 1;

        ctx.fillStyle = '#ffffff';
        ctx.fillRect(xPixel * this.PIXEL_WIDTH, yPixel * this.PIXEL_WIDTH, 
            this.PIXEL_WIDTH, this.PIXEL_WIDTH);
    },

    train: function() {
        var digitVal = document.getElementById("digit").value;
        if (!digitVal || this.data.indexOf(1) < 0) {
            alert("Please type and draw a digit value in order to train the network");
            return;
        }
        this.trainArray.push({"y0": this.data.slice(), "label": parseInt(digitVal)});
        this.trainingRequestCount++;

        // Time to send a training batch to the server.
        if (this.trainingRequestCount === this.BATCH_SIZE) {
            alert("Sending training data to server...");
            var json = {
                trainArray: this.trainArray,
                train: true
            };

            this.sendData(json);
            this.trainingRequestCount = 0;
            this.trainArray = [];
            this.data = new Array(this.TRANSLATED_WIDTH * this.TRANSLATED_WIDTH).fill(0);
        }
    },

    test: function() {
        if (this.data.indexOf(1) < 0) {
            alert("Please draw a digit in order to test the network");
            return;
        }
        var json = {
            image: this.data,
            predict: true
        };
        this.sendData(json);
    },

    receiveResponse: function(xmlHttp) {
        if (xmlHttp.status !== 200) {
            alert("Server returned status " + xmlHttp.status);
            return;
        }
        var responseJSON = JSON.parse(xmlHttp.responseText);
        if (responseJSON.type === "test") {
            alert("The neural network predicts you wrote a '" + responseJSON.result + "'");
        }
    },

    onError: function(e) {
        alert("Error occurred while connecting to server: " + e.target.statusText);
    },

    sendData: function(json) {
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.open('POST', this.HOST + ":" + this.PORT, true);
        xmlHttp.onload = function() { this.receiveResponse(xmlHttp); }.bind(this);
        xmlHttp.onerror = function() { this.onError(xmlHttp); }.bind(this);
        var msg = JSON.stringify(json);
        xmlHttp.setRequestHeader('Content-Type', 'application/json');
        xmlHttp.send(msg);
    }
};
