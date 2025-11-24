document.getElementById("signupForm").addEventListener("submit", async function (event) {
    event.preventDefault();

    let name = document.getElementById("name").value;
    let email = document.getElementById("email").value;
    let messageEl = document.getElementById("message");

    let response = await fetch("http://127.0.0.1:8000/signup/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email }),
    });

    let data = await response.json();
    messageEl.textContent = data.message || "Signup failed!";
});
