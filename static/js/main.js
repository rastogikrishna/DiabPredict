// DiabPredict UI Behaviors

document.addEventListener('DOMContentLoaded', function() {
    // 1. Smoothly fill SHAP contribution bars on page render
    const shapBars = document.querySelectorAll('.shap-bar');
    shapBars.forEach(bar => {
        const targetWidth = bar.getAttribute('data-width');
        setTimeout(() => {
            bar.style.width = targetWidth + '%';
        }, 200);
    });

    // 2. Add validation indicators to prediction forms
    const forms = document.querySelectorAll('.needs-validation');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
});
