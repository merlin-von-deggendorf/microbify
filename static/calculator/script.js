document.addEventListener('DOMContentLoaded', () => {
    main();
});

function main() {
    // Get references to all input fields using the updated IDs
    const targetVolume = document.getElementById('targetVolume');
    const od600 = document.getElementById('od600');
    const conversionFactor = document.getElementById('conversionFactor');
    const targetCellCount = document.getElementById('targetCellCount');
    const dryMassConversion = document.getElementById('dryMassConversion');
    const targetDryMass = document.getElementById('targetDryMass');

    // Example: log all references
    console.log("Input field references:", { 
        targetVolume, 
        od600, 
        conversionFactor, 
        targetCellCount, 
        dryMassConversion, 
        targetDryMass 
    });
    
    // Your additional calculation code goes here.
    console.log("DOM fully loaded and parsed. Running main...");
}