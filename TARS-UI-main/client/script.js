import bot from './assets/bot.svg'
import user from './assets/user.svg'
window.onload = function() {
  let ip = document.querySelector('#iparea').value;
  let key = document.querySelector('#keyarea').value;
  console.log("IP ADDRESS:" + ip)
  const form = document.querySelector('form')
  const chatContainer = document.querySelector('#chat_container')
  let loadInterval
  function loader(element) {
      element.textContent = ''

      loadInterval = setInterval(() => {
          // Update the text content of the loading indicator
          element.textContent += '.';

          // If the loading indicator has reached three dots, reset it
          if (element.textContent === '....') {
              element.textContent = '';
          }
      }, 300);
  }

  function typeText(element, text) {
      console.log("typing text")
      console.log(text)
      let index = 0;
      const cursor = document.createElement('span'); // Create cursor element
      cursor.classList.add('cursor');
      element.appendChild(cursor); // Append cursor to element
      cursor.style.left = '0'; // Set initial cursor position
      const words = text.split(' '); // Split text into array of words
      const interval = setInterval(() => {
        if (index < words.length) {
          const word = words[index];
          const span = document.createElement('span'); // Create span element for each character
          span.textContent = word + ' '; // Add space after each word
          element.insertBefore(span, cursor); // Insert span before cursor
          index++;
          element.parentElement.parentElement.parentElement.scrollTop = element.parentElement.parentElement.parentElement.scrollHeight;
        } else {
          clearInterval(interval);
          setTimeout(() => {
              element.removeChild(cursor); // Remove cursor from element
              element.classList.remove('cursor')
            }, 450); // Wait for 0.5 seconds before removing the cursor
        }
      }, 60);
    }

  // generate unique ID for each message div of bot
  // necessary for typing text effect for that specific reply
  // without unique ID, typing text will work on every element
  function generateUniqueId() {
      const timestamp = Date.now();
      const randomNumber = Math.random();
      const hexadecimalString = randomNumber.toString(16);

      return `id-${timestamp}-${hexadecimalString}`;
  }

  function chatStripe(isAi, value, uniqueId) {
      return (
          `
          <div class="wrapper ${isAi && 'ai'}">
              <div class="chat">
                  <div class="profile">
                      <img 
                        src=${isAi ? bot : user} 
                        alt="${isAi ? 'bot' : 'user'}" 
                      />
                  </div>
                  <div class="message" id=${uniqueId}>${value}</div>
              </div>
          </div>
      `
      )
  }
      // bot's chatstripe
  const uniqueId = generateUniqueId()
  chatContainer.innerHTML += chatStripe(true, "", uniqueId)
  // specific message div 
  const greetingDiv = document.getElementById(uniqueId)
  loader(greetingDiv)
  fetch('http://' + ip + ':8000/chat', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'Authorization': key
    },
    body: JSON.stringify({
      prompt: "hi there"
    })
  })
  .then(response => {
    console.log(response)
    if (response.ok) {
      console.log("Response OK")
      return response.json();
    } else {
      console.log("Response not OK")
      throw new Error(response.statusText);
    }
  })
  .then(data => {
    console.log(data)
    clearInterval(loadInterval)
    greetingDiv.innerHTML = ""
    const parsedData = data.response.trim(); // trims any trailing spaces/'\n' 
    console.log("Parsed data")
    console.log(parsedData)
    typeText(greetingDiv, parsedData)
    form.querySelector('textarea').removeAttribute('disabled');
  })
  .catch(error => {
    console.log(error)
    clearInterval(loadInterval)
    greetingDiv.innerHTML = ""
    greetingDiv.innerHTML = "There was an error connecting to TARS. Are you authenticated?"
    greetingDiv.style.color = '#c0392b'; // Set the text color to red
    form.querySelector('textarea').removeAttribute('disabled');
    console.error(error);
  });


  const handleSubmit = async (e) => {
      let ip = document.querySelector('#iparea').value;
      let key = document.querySelector('#keyarea').value;
      console.log(ip)
      e.preventDefault()
      const data = new FormData(form)

      // user's chatstripe
      chatContainer.innerHTML += chatStripe(false, "" + data.get('prompt'))

      // to clear the textarea input 
      form.reset()

      // bot's chatstripe
      const uniqueId = generateUniqueId()
      chatContainer.innerHTML += chatStripe(true, "", uniqueId)

      // to focus scroll to the bottom 
      chatContainer.scrollTop = chatContainer.scrollHeight;

      // specific message div 
      const messageDiv = document.getElementById(uniqueId)
      form.querySelector('textarea').setAttribute('disabled', true); 
      // messageDiv.innerHTML = "..."
      loader(messageDiv)
      console.log("Fetching data")
      console.log(data.get('prompt'))
      fetch('http://' + ip + ':8000/chat', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'Authorization': key
      },
      body: JSON.stringify({
        prompt: data.get('prompt')
      })
    })
    .then(response => {
      console.log(response)
      if (response.ok) {
        console.log("Response OK")
        return response.json();
      } else {
        console.log("Response not OK")
        throw new Error(response.statusText);
      }
    })
    .then(data => {
      console.log(data)
      clearInterval(loadInterval)
      messageDiv.innerHTML = ""
      const parsedData = data.response.trim(); // trims any trailing spaces/'\n' 
      console.log("Parsed data")
      console.log(parsedData)
      typeText(messageDiv, parsedData)
      form.querySelector('textarea').removeAttribute('disabled');
    })
    .catch(error => {
      console.log(error)
      clearInterval(loadInterval)
      messageDiv.innerHTML = ""
      
      messageDiv.innerHTML = "An error occured. Please check your authentication, and TARS' status."
      messageDiv.style.color = '#c0392b'; // Set the text color to red
      form.querySelector('textarea').removeAttribute('disabled');
      console.error(error);
    });
  }

  form.addEventListener('submit', handleSubmit)
  form.addEventListener('keyup', (e) => {
      if (e.keyCode === 13) {
          handleSubmit(e)
      }
  })
}