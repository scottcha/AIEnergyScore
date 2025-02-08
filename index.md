<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Energy Score Documentation & FAQ</title>
  <!-- Polyfill and MathJax Scripts -->
  <script type="text/javascript" async src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
  <script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js"></script>

  <style>
    /* --- Sticky Navigation Bar --- */
    nav {
      position: sticky;
      top: 0;
      z-index: 1000;
      display: flex;
      justify-content: center; /* Centers the links */
      gap: 40px; /* Adds space between links */
      background-color: white; /* Background color */
      padding: 10px 0; /* Padding for spacing */
      border-bottom: 1px solid #ccc; /* Visual divider */
    }

    nav a {
      text-decoration: none; /* Removes underline */
      color: black; /* Makes links black */
      font-size: 16px; /* Increases font size */
      font-weight: bold; /* Bold text */
    }

    nav a:hover,
    nav a.active { /* Highlight when hovered or active (scrollspy) */
      text-decoration: underline;
    }

    /* --- Hide GitHub Pages Banner --- */
    body > header, 
    .brand, 
    .site-header, 
    .page-header, 
    .github-pages-metadata { 
      display: none !important; 
    }

    .container-lg > div:first-child {
      display: none !important;
    }

    /* --- Interactive TOC Styles --- */
    #toc {
      margin: 20px auto;
      max-width: 800px;
      padding: 10px;
      background: #f9f9f9;
      border: 1px solid #ddd;
    }
    #toc h2 {
      margin-top: 0;
    }
    #toc a {
      display: block;
      padding: 5px;
      color: #333;
      text-decoration: none;
    }
    #toc a.active {
      background-color: #ddd;
    }

    /* --- Collapsible FAQ (Accordion) Styles --- */
    details {
      margin: 10px 0;
      border: 1px solid #ddd;
      padding: 10px;
      border-radius: 4px;
    }
    summary {
      font-weight: bold;
      cursor: pointer;
      outline: none;
    }

    /* Optional: Style for the main content sections */
    section {
      max-width: 800px;
      margin: 40px auto;
      padding: 0 10px;
    }
  </style>
</head>
<body>

  <!-- Sticky Navigation Bar -->
  <nav>
    <a href="https://huggingface.co/spaces/AIEnergyScore/Leaderboard">Leaderboard</a>
    <a href="https://huggingface.co/spaces/AIEnergyScore/submission_portal">Submission Portal</a>
    <a href="https://huggingface.co/spaces/AIEnergyScore/README/discussions">Community</a>
    <a href="#faq">FAQ</a>
    <a href="#documentation">Documentation</a>
  </nav>

  <p align="center">
    <img src="https://raw.githubusercontent.com/huggingface/AIEnergyScore/refs/heads/main/logo.png" alt="AI Energy Score" width="300">
  </p>

  <!-- Interactive Table of Contents -->
  <div id="toc">
    <h2>Table of Contents</h2>
    <a href="#faq">FAQ</a>
    <a href="#documentation">Documentation</a>
  </div>

  <!-- FAQ Section with Collapsible Panels -->
  <section id="faq">
    <h2>FAQ</h2>
    
    <details>
      <summary>What is the goal of this project?</summary>
      <p>
        The goal of AI Energy Score is to establish a standardized approach for evaluating the energy efficiency of AI model inference. By focusing on controlled and comparable metrics, such as specific tasks and hardware, we aim to provide useful insights for researchers, developers, organizations, and policymakers.
      </p>
    </details>

    <details>
      <summary>What do the star ratings mean?</summary>
      <p>
        The star ratings represent the relative energy efficiency of an AI model for a specific task on a particular leaderboard. A 5-star rating means the model is among the most energy efficient, while a 1-star rating indicates the opposite. The ratings are recalibrated with each leaderboard update.
      </p>
    </details>

    <!-- Add more FAQ details elements as needed -->
  </section>

  <!-- Documentation Section -->
  <section id="documentation">
    <h2>Documentation</h2>
    <p>
      Welcome to the documentation section. Here you can find detailed information on methodology, experimental setup, and further aspects of the AI Energy Score project.
    </p>
    <!-- Include your full documentation content here -->
  </section>

  <!-- JavaScript for Interactive TOC (Scrollspy) -->
  <script>
    // Wait for the DOM to load
    document.addEventListener("DOMContentLoaded", function() {
      const tocLinks = document.querySelectorAll("#toc a");
      const sections = document.querySelectorAll("section");

      // Function to update active TOC link based on scroll position
      function onScroll() {
        const scrollPos = document.documentElement.scrollTop || document.body.scrollTop;
        sections.forEach(section => {
          // Add an offset (e.g., 50px) for better accuracy with sticky nav height
          if (section.offsetTop <= scrollPos + 50 && (section.offsetTop + section.offsetHeight) > scrollPos + 50) {
            tocLinks.forEach(link => {
              link.classList.remove("active");
              if (section.getAttribute("id") === link.getAttribute("href").substring(1)) {
                link.classList.add("active");
              }
            });
          }
        });
      }

      // Listen to scroll events
      window.addEventListener("scroll", onScroll);
      // Initial call to set the active link on page load
      onScroll();
    });
  </script>

</body>
</html>
