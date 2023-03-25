var answers = ["The_Scream.jpg",
    "The_Scream.jpg",
    "Kand1.jpeg",
    "Kand2.png",
    "Monet.jpg",
    "Picasso.png",
    "VanGogh.png"
];

function choose(choices) {
    var index = Math.floor(Math.random() * choices.length);
    return choices[index];
}

window.onload = loadImageInCanvas("VanGogh.png", document.getElementById('style_img'))

document.getElementById('style_choice').onchange = function(e) {
    loadImageInCanvas(document.getElementById("style_choice").value, document.getElementById('style_img'));
};

AWS.config.region = 'eu-west-1'; // Region

AWS.config.credentials = new AWS.CognitoIdentityCredentials({
    IdentityPoolId: 'eu-west-1:daac3c5a-13e3-4c7d-80d8-869eacaa0f83',
});

var bucketName = 'visualneurons.com-videos'; // Enter your bucket name
var bucket = new AWS.S3({
    params: {
        Bucket: bucketName
    }
});

var fileChooser = document.getElementById('file-chooser');
var button = document.getElementById('upload-button');
var results = document.getElementById('results');

fileChooser.onchange = function(e) {
    var file = this.files[0]; // Get uploaded file
    validateFile(file) // Validate Duration
}

function validateFile(file) {

    var video = document.createElement('video');
    video.preload = 'metadata';

    video.onloadedmetadata = function() {

        window.URL.revokeObjectURL(video.src);

        if (video.duration > 30) {
            results.textContent = "Your videos exceeds our current limit of 30s. Sorry, this is too long for us!"
            setTimeout(function() {
                results.textContent = ''
            }, 5000);
            fileChooser.value = '';
        }

    }

    video.src = URL.createObjectURL(file);
}

function validateEmail(email) {
    var re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

button.addEventListener('click', function() {
    email = document.getElementById('email').value.toLowerCase();

    if (validateEmail(email) == false) {
        results.textContent = "Please enter a valid email address and try again!"
        setTimeout(function() {
            results.textContent = ''
        }, 2000);
        return
    }

    var file = fileChooser.files[0];

    if (file) {
        this.disabled = true;
        button.style.backgroundColor = "#ffc477"
        button.textContent = "Uploaded"

        results.textContent = '';
        clean_email = email.replace(/[^a-zA-Z0-9]/g, '')
        var objKey = clean_email + '_' + file.name.replace(/[^a-zA-Z0-9]/g, '').replace(' ', '').replace('gif', '.gif');
        var params = {
            Key: objKey,
            ContentType: file.type,
            Body: file,
            Metadata: {
                'email': email,
                'style': document.getElementById("style_choice").value,
            },
        };

        bucket.putObject(params, function(err, data) {
            if (err) {
                results.textContent = 'ERROR: ' + err;
            } else {
                results.textContent = "GIF ingested successfully!";
            }
        });
    } else {
        results.textContent = 'Nothing to upload.';
    }
}, false);

function loadImageInCanvas(url, canvas) {
    var img = $("<img />", {
        src: url,
        crossOrigin: "Anonymous",
    }).load(draw).error(failed);

    function draw() {
        canvas.width = this.width * (300 / this.height);
        canvas.height = 300;
        var ctx = canvas.getContext('2d');
        ctx.drawImage(this, 0, 0, this.width, this.height, 0, 0, this.width * (300 / this.height), 300);
    }

    function failed() {
        alert("The provided file couldn't be loaded as an Image media");
    };

}