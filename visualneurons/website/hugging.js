var API_ENDPOINT = "https:50vllx1cci.execute-api.eu-west-1.amazonaws.com/prod"
var prompt_ = document.getElementById('hugging-prompt');
var samples = document.getElementById('how-many-samples');
var words = document.getElementById('how-many-words');
var temperature = document.getElementById('temperature');
var nucleus = document.getElementById('nucleus');
var topn = document.getElementById('top-n');
var button = document.getElementById('hugging');

function truncatePrompt(prompt){
    prompt = prompt.value + " "
    prompt = prompt.substring(0,100)
    index = prompt.lastIndexOf(" ")
    return prompt.substring(0, index)
}

function validate(str, min, max) {
    n = parseFloat(str);
    return (!isNaN(n) && n >= min && n <= max);
  }

function processResponse(response){
    msg = JSON.parse(response)
    console.log(msg)
    addResponse(msg)
    button.disabled = !button.disabled;
    button.style.backgroundColor = "#FF9900"
    button.textContent = "Let the machine take over!"
}

function addResponse(msg) {
    document.getElementById("results").textContent = ""
    var para = document.createElement("div");
    var s = "<div class='center' id='gpt2'> <b> Here what GPT-2 has to say... </b> </br></br>"
    i = 1
    for (var key in msg){
        var value = msg[key];
        s = s + i + ") " + prompt + " <b>" + value + "</b> </br></br>"
        i = i + 1
    }
    para.innerHTML = s + "</div>";
    document.getElementById('append').appendChild(para);
  }

button.addEventListener('click', function() {

    if(!validate(samples.value, 1, 4)) {
        alert("The number of text samples must be between 1 and 4. You have selected ".concat(samples.value, "!"));
        return false;
    }
    if(!validate(words.value, 1, 100)) {
        alert("The number of words must be between 1 and 100. You have selected ".concat(words.value, "!"));
        return false;
    }
    if(!validate(temperature.value, 0, 1)) {
        alert("Temperature must be between 0 and 1. You have selected ".concat(temperature.value, "!"));
        return false;
    }
    if(!validate(nucleus.value, 0, 1)) {
        alert("A probability must be between 0 and 1. In nucleus filtering you have typed ".concat(nucleus.value, "!"));
        return false;
    }
    if(!validate(topn.value, 0, 1000)) {
        alert("You can select between 1 and 1000 top N words. You have typed ".concat(topn.value, "!"));
        return false;
    }
    if(prompt_.value.length==0){
        alert("Your text prompt is empty! Please trigger the model with at least one word.");
        return false
    }

    prompt = truncatePrompt(prompt_)
    var inputData = {"prompt": prompt,
                    "num_samples": samples.value,
                    "length": words.value,
                    "temperature": temperature.value,
                    "top_p": nucleus.value,
                    "top_k": topn.value
                }

    console.log(inputData)
    button.disabled = true;
    button.style.backgroundColor = "#ffc477"
    button.textContent = "Running"

    element = document.getElementById('gpt2') 
    if(element!=null){element.parentNode.removeChild(element)}
    document.getElementById("results").textContent = "Hold on, checking if we got some deep writers around..."

    $.ajax({
        url: API_ENDPOINT,
        type: 'POST',
        crossDomain: true,
        tryCount : 0,
        retryLimit : 15,
        dataType: 'json',
        contentType: "application/json",
        data: JSON.stringify(inputData),
        success: processResponse,
        error: function(xhr, status, error) {
            document.getElementById("results").textContent = "Ouch... Sorry, we have disabled our deep neural writers for the time being!";            
            return;
            console.log("AJAX status:" + status)
            console.log("retry " + this.tryCount + " of " + this.retryLimit)
            if (status == 'error') {
                this.tryCount++;
                if (this.tryCount <= this.retryLimit) {
                    //try again
                    $.ajax(this);
                    if (this.tryCount==1){document.getElementById("results").textContent = "Found them! Stretching their fingers..."}
                    if (this.tryCount==2){document.getElementById("results").textContent = "Finding inspiration..."}
                    if (this.tryCount==3){document.getElementById("results").textContent = "It is hard to come up with something cool..."}
                    if (this.tryCount==4){document.getElementById("results").textContent = "Don't give up! We are almost there..."}
                    if (this.tryCount==5){document.getElementById("results").textContent = "Ok, I admit they are being slow..."}
                    if (this.tryCount==6){document.getElementById("results").textContent = "I hear some typing!"}
                    if (this.tryCount>=7){document.getElementById("results").textContent = "Still nothing?"}
                    return;
                }
                document.getElementById("results").textContent = "Ouch... Sorry, it seems we ran out of deep neural writers! Can you try again in a couple of minutes?";            
                return;
            }
            
        }
    });
}, false);