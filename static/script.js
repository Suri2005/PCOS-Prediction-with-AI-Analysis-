/* ==========================================================================
   PCOS PREDICT CLIENT-SIDE INTERACTIVE ENGINE
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Hamburger Menu Toggle
  initHamburgerMenu();

  // 2. Live BMI Calculator
  initLiveBMICalculator();

  // 3. Symptom Severity Slider Interactivity
  initSeveritySliders();

  // 4. Progress Bars Animation (Result Page)
  initProgressBarAnimation();

  // 5. Custom Risk Gauge Animation (Result Page)
  initRiskGaugeAnimation();

  // 6. Client-Side Form Validation & Loading Spinner
  initFormValidationAndLoader();

  // 7. Flash Messages Auto-Dismiss
  initFlashMessages();

  // 8. Smooth Scrolling with Offset
  initSmoothScroll();
});

/**
 * 1. Hamburger Menu Toggle (Mobile Menu Drawers)
 */
function initHamburgerMenu() {
  const menuBtn = document.getElementById('menu-btn');
  const navLinksMenu = document.getElementById('nav-links-menu');

  if (!menuBtn || !navLinksMenu) return;

  menuBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    navLinksMenu.classList.toggle('open');
  });

  // Close menu when clicking outside of navbar
  document.addEventListener('click', (e) => {
    if (!navLinksMenu.contains(e.target) && !menuBtn.contains(e.target)) {
      navLinksMenu.classList.remove('open');
    }
  });

  // Close menu on window resize if scaling to desktop dimensions
  window.addEventListener('resize', () => {
    if (window.innerWidth >= 1024) {
      navLinksMenu.classList.remove('open');
    }
  });
}

/**
 * 2. Live BMI Calculator
 */
function initLiveBMICalculator() {
  const heightInput = document.getElementById('height');
  const weightInput = document.getElementById('weight');
  const bmiDisplay = document.getElementById('live-bmi');
  const bmiStatusLabel = document.getElementById('live-bmi-status');

  if (!heightInput || !weightInput || !bmiDisplay || !bmiStatusLabel) return;

  function calculateBMI() {
    const height = parseFloat(heightInput.value);
    const weight = parseFloat(weightInput.value);

    if (isNaN(height) || isNaN(weight) || height <= 0 || weight <= 0) {
      bmiDisplay.textContent = '--';
      bmiStatusLabel.textContent = 'Enter details';
      bmiStatusLabel.className = 'bmi-status-label badge badge-low';
      return;
    }

    const heightInMeters = height / 100.0;
    const bmi = weight / (heightInMeters * heightInMeters);
    const formattedBmi = bmi.toFixed(1);
    
    bmiDisplay.textContent = formattedBmi;

    // Apply class list styling based on WHO weight classifications
    bmiStatusLabel.className = 'bmi-status-label badge ';
    if (bmi < 18.5) {
      bmiStatusLabel.textContent = 'Underweight';
      bmiStatusLabel.classList.add('badge-medium'); // Amber-colored alert
    } else if (bmi >= 18.5 && bmi < 25.0) {
      bmiStatusLabel.textContent = 'Normal';
      bmiStatusLabel.classList.add('badge-low'); // Green-colored normal
    } else if (bmi >= 25.0 && bmi < 30.0) {
      bmiStatusLabel.textContent = 'Overweight';
      bmiStatusLabel.classList.add('badge-medium'); // Amber alert
    } else {
      bmiStatusLabel.textContent = 'Obese';
      bmiStatusLabel.classList.add('badge-high'); // Red alert
    }
  }

  // Attach keyup and input listeners for real-time keystroke tracking
  heightInput.addEventListener('input', calculateBMI);
  weightInput.addEventListener('input', calculateBMI);
}

/**
 * 3. Symptom Severity Sliders
 */
function initSeveritySliders() {
  const sliders = document.querySelectorAll('.range-slider');
  
  sliders.forEach(slider => {
    const valDisplay = slider.parentElement.querySelector('.slider-val-num');
    const badge = slider.parentElement.parentElement.querySelector('.slider-badge');

    function updateSliderState() {
      const val = parseInt(slider.value);
      if (valDisplay) valDisplay.textContent = val;

      if (badge) {
        // Clear classes
        badge.className = 'slider-badge badge';

        // Update text labels and apply severity color scheme
        if (val === 1) {
          badge.textContent = 'None';
          badge.classList.add('badge-low');
        } else if (val === 2) {
          badge.textContent = 'Mild';
          badge.classList.add('badge-low');
        } else if (val === 3) {
          badge.textContent = 'Moderate';
          badge.classList.add('badge-medium');
        } else if (val === 4) {
          badge.textContent = 'Severe';
          badge.classList.add('badge-high');
        } else if (val === 5) {
          badge.textContent = 'Very Severe';
          badge.classList.add('badge-high');
        }
      }

      // Update slider track background color
      const pct = ((val - 1) / 4) * 100;
      let activeColor = 'var(--pink)';
      if (val <= 2) {
        activeColor = 'var(--green)';
      } else if (val === 3) {
        activeColor = 'var(--amber)';
      } else {
        activeColor = 'var(--red)';
      }
      slider.style.background = `linear-gradient(to right, ${activeColor} ${pct}%, #ddd ${pct}%)`;
    }

    slider.addEventListener('input', updateSliderState);
    updateSliderState(); // run once on initialization
  });
}

/**
 * 4. Progress Bars Animation (Result Page)
 */
function initProgressBarAnimation() {
  const groups = document.querySelectorAll('.indicator-group');
  if (groups.length === 0) return;

  // Small timeout to let elements render in DOM, then stagger widths
  setTimeout(() => {
    groups.forEach((group, index) => {
      const val = parseInt(group.getAttribute('data-value')) || 0;
      const fill = group.querySelector('.bar-fill');
      if (fill) {
        setTimeout(() => {
          fill.style.width = `${val}%`;
        }, index * 150); // Stagger by 150ms per bar
      }
    });
  }, 100);
}

/**
 * 5. Custom Risk Gauge Animation (Result Page)
 */
function initRiskGaugeAnimation() {
  const fill = document.getElementById('gauge-fill');
  const needle = document.getElementById('gauge-needle');

  if (!fill || !needle) return;

  const risk = parseFloat(fill.getAttribute('data-risk')) || 0;

  // Semicircle rotates 180 degrees max.
  // Needle points left (-90deg) at 0% and right (+90deg) at 100%.
  const needleRotation = -90 + (risk / 100) * 180;
  const fillRotation = (risk / 100) * 180;

  // Let the browser load the transition parameters, then trigger transform
  setTimeout(() => {
    fill.style.transform = `rotate(${fillRotation}deg)`;
    needle.style.transform = `translateX(-50%) rotate(${needleRotation}deg)`;
  }, 200);
}

/**
 * 6. Client-Side Form Validation & Loading Spinner
 */
function initFormValidationAndLoader() {
  const form = document.getElementById('pcos-form');
  if (!form) return;

  form.addEventListener('submit', (e) => {
    // Clear existing validation messages
    document.querySelectorAll('.inline-validation-error').forEach(el => el.remove());

    let isValid = true;
    let firstError = null;

    const nameInput = document.getElementById('name');
    const ageInput = document.getElementById('age');
    const heightInput = document.getElementById('height');
    const weightInput = document.getElementById('weight');
    const emailInput = document.getElementById('email');

    function raiseError(inputElement, msg) {
      isValid = false;
      const errorMsg = document.createElement('span');
      errorMsg.className = 'inline-validation-error';
      errorMsg.style.color = 'var(--red)';
      errorMsg.style.fontSize = '0.8rem';
      errorMsg.style.fontWeight = '700';
      errorMsg.style.marginTop = '4px';
      errorMsg.textContent = msg;

      inputElement.parentElement.appendChild(errorMsg);
      
      if (!firstError) {
        firstError = inputElement;
      }
    }

    // Name Validation
    if (!nameInput.value.trim()) {
      raiseError(nameInput, 'Full name is required.');
    }

    // Age Validation
    const age = parseInt(ageInput.value);
    if (isNaN(age) || age < 13 || age > 55) {
      raiseError(ageInput, 'Age must be a number between 13 and 55.');
    }

    // Height Validation
    const height = parseFloat(heightInput.value);
    if (isNaN(height) || height < 100 || height > 220) {
      raiseError(heightInput, 'Height must be between 100 and 220 cm.');
    }

    // Weight Validation
    const weight = parseFloat(weightInput.value);
    if (isNaN(weight) || weight < 30 || weight > 200) {
      raiseError(weightInput, 'Weight must be between 30 and 200 kg.');
    }

    // Email Validation
    const email = emailInput.value.trim();
    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!email || !emailPattern.test(email)) {
      raiseError(emailInput, 'Please enter a valid email address.');
    }

    if (!isValid) {
      e.preventDefault(); // Stop form submission
      if (firstError) {
        firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
        firstError.focus();
      }
    } else {
      // Toggle button spinner
      const submitBtn = document.getElementById('submit-btn');
      const btnText = document.getElementById('btn-text');
      const btnSpinner = document.getElementById('btn-spinner');
      const btnLoadingText = document.getElementById('btn-loading-text');

      if (submitBtn && btnText && btnSpinner && btnLoadingText) {
        btnText.style.display = 'none';
        btnSpinner.style.display = 'inline-block';
        btnLoadingText.style.display = 'inline-block';
        submitBtn.disabled = true;
        submitBtn.style.opacity = '0.85';
        submitBtn.style.cursor = 'not-allowed';
      }
    }
  });
}

/**
 * 7. Flash Messages Toggles & Auto-Dismissal
 */
function initFlashMessages() {
  const flashMessages = document.querySelectorAll('.flash-message');

  flashMessages.forEach(flash => {
    const closeBtn = flash.querySelector('.flash-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => dismiss(flash));
    }

    // Auto-dismiss after 4 seconds (4000ms)
    setTimeout(() => {
      dismiss(flash);
    }, 4000);
  });

  function dismiss(element) {
    element.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
    element.style.opacity = '0';
    element.style.transform = 'translateY(-8px)';
    setTimeout(() => {
      element.remove();
    }, 400);
  }
}

/**
 * 8. Smooth Scrolling with Offset
 */
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const targetId = this.getAttribute('href');
      if (targetId === '#') return;

      const target = document.querySelector(targetId);
      if (target) {
        e.preventDefault();
        const navHeight = 70; // 70px offset for the fixed header
        const targetPos = target.getBoundingClientRect().top + window.pageYOffset - navHeight;

        window.scrollTo({
          top: targetPos,
          behavior: 'smooth'
        });
      }
    });
  });
}
