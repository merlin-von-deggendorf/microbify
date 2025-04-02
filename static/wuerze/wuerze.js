document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementsByClassName('content')[0];

    const form = document.getElementById('stammwuerze-form');
    const result = document.getElementById('result');

    form.addEventListener('submit', (event) => {
        event.preventDefault();
        const density = parseFloat(document.getElementById('density').value);
        if (!isNaN(density)) {
            const gradPlato = (-616.868) + (1111.14 * density) - (630.272 * Math.pow(density, 2)) + (135.997 * Math.pow(density, 3));
            result.innerHTML = `
                <p>Berechnung:</p>
                <p>-616.868 + (1111.14 × ${density}) - (630.272 × ${density}²) + (135.997 × ${density}³)</p>
                <p>Stammwürze: ${gradPlato.toFixed(2)} °P</p>
            `;

            // Check if the calculated value is within the range for any beer type
            const rows = document.querySelectorAll('#stammwuerze-table tbody tr');
            rows.forEach(row => {
                row.classList.remove('highlight'); // Remove any existing highlight
                const min = parseFloat(row.cells[1].textContent) || null;
                const max = parseFloat(row.cells[2].textContent) || null;

                if ((min === null || gradPlato >= min) && (max === null || gradPlato <= max)) {
                    row.classList.add('highlight'); // Add highlight class
                }
            });
        } else {
            result.textContent = 'Bitte geben Sie eine gültige spezifische Dichte ein.';
        }
    });

    const resetButton = document.getElementById('reset-button');

    resetButton.addEventListener('click', () => {
        document.getElementById('density').value = '1.050'; // Reset SG to 1.050
        result.textContent = '';

        // Reset the background color of all table rows
        const rows = document.querySelectorAll('#stammwuerze-table tbody tr');
        rows.forEach(row => {
            row.classList.remove('highlight'); // Remove highlight class
        });
    });
});
