var API_ENDPOINT = "https://afwdigg7c8.execute-api.eu-west-1.amazonaws.com/api/"

document.getElementById('inp').onchange = function(e) {
    var img = new Image();
    img.onload = draw;
    img.onerror = failed;
    img.src = URL.createObjectURL(this.files[0]);
  };
  function draw() {
    var canvas = document.getElementById('canvas');
    canvas.width = this.width;
    canvas.height = this.height;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(this, 0,0);
  }
  function failed() {
    console.error("The provided file couldn't be loaded as an Image media");
};

document.getElementById("genre").onclick = function(){
    var canvas = document.getElementById("canvas")
    var image = canvas.toDataURL()
	var inputData = {"data": image};

	$.ajax({
	      url: API_ENDPOINT,
        type: 'POST',
	      data:  JSON.stringify(inputData)  ,
        contentType: 'application/json; charset=utf-8',
        //dataType: 'jsonp',
	      success: function (response) {
					document.getElementById("genreReturned").textContent="Prediction: " + response;
	      },
	      error: function () {
	          alert("error");
	      }
	  });
}