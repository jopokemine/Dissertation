'use strict';

const sendBtnClickListener = async () => {
    const msg = document.getElementById('msgTxt').value

    const response = await fetch('/chatbot', {
        method: 'POST',
        mode: 'same-origin',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            msg
        })
    });
    if (response.ok) {
        const data = await response.json();
        // console.log(await data);
        document.getElementById("botMsg").textContent = data;
    }
}

const init = () => {
    document.getElementById('sendBtn').addEventListener('click', sendBtnClickListener);
}

window.addEventListener('load', init)
