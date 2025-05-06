document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("searchInput");
    const suggestionsBox = document.getElementById("suggestions");
  
    let profileNames = [];
  
    fetch("/static/profile_names.json")
      .then(response => response.json())
      .then(data => profileNames = data)
      .catch(err => console.error("Failed to load suggestions:", err));
  
    input.addEventListener("input", () => {
      const query = input.value.toLowerCase();
      suggestionsBox.innerHTML = "";
  
      if (query.length === 0) {
        suggestionsBox.style.display = "none";
        return;
      }
  
      const matches = profileNames.filter(name =>
        name.toLowerCase().includes(query)
      ).slice(0, 5); // Show top 5
  
      if (matches.length === 0) {
        suggestionsBox.style.display = "none";
        return;
      }
  
      matches.forEach(match => {
        const div = document.createElement("div");
        div.classList.add("suggestion-item");
        div.textContent = match;
        div.onclick = () => {
          input.value = match;
          suggestionsBox.innerHTML = "";
          suggestionsBox.style.display = "none";
        };
        suggestionsBox.appendChild(div);
      });
  
      suggestionsBox.style.display = "block";
    });
  
    document.addEventListener("click", (e) => {
      if (!suggestionsBox.contains(e.target) && e.target !== input) {
        suggestionsBox.innerHTML = "";
        suggestionsBox.style.display = "none";
      }
    });
  });
  
  
  