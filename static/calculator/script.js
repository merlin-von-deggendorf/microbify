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
    const calculateButton = document.getElementById('calculateBtn');
    // Get references for M9 Medium Components
    const Na2HPO4 = document.getElementById('Na2HPO4');
    const KH2PO4 = document.getElementById('KH2PO4');
    const NaCl = document.getElementById('NaCl');
    const NH4Cl = document.getElementById('NH4Cl');
    const MgSO4 = document.getElementById('MgSO4');
    const CaCl2 = document.getElementById('CaCl2');
    const Glucose = document.getElementById('Glucose');

    calculateButton.addEventListener('click', () => {
        const volume = parseFloat(targetVolume.value); // in mL
        const od = parseFloat(od600.value);              // unitless
        const factor = parseFloat(conversionFactor.value); // cells/mL per OD unit
        
        if (isNaN(volume) || isNaN(od) || isNaN(factor)) {
            alert("Please enter valid numbers for Target Volume, OD600, and Conversion Factor.");
            return;
        }
        
        // Calculate Total Cell Count (cells)
        const cellCount = volume * od * factor;
        targetCellCount.value = cellCount;
        
        // Calculate Total Dry Mass (in grams)
        // dryMassConversion is given in picograms per cell; 1 g = 1e12 picograms
        const massConv = parseFloat(dryMassConversion.value); // picograms per cell
        const dryMass = cellCount * massConv / 1e12;
        targetDryMass.value = dryMass;
        
        // Calculate M9 Medium Components based on dryMass (per g dry biomass)
        // Stoichiometric coefficients (g component per g biomass):
        // Na₂HPO₄: 0.073, KH₂PO₄: 0.037, NaCl: 0.083, NH₄Cl: 0.54,
        // MgSO₄·7H₂O: 0.04, CaCl₂·2H₂O: 0.002, Glucose: 2.0
        Na2HPO4.value = (0.073 * dryMass).toFixed(3);
        KH2PO4.value = (0.037 * dryMass).toFixed(3);
        NaCl.value = (0.083 * dryMass).toFixed(3);
        NH4Cl.value = (0.54 * dryMass).toFixed(3);
        MgSO4.value = (0.04 * dryMass).toFixed(3);
        CaCl2.value = (0.002 * dryMass).toFixed(3);
        Glucose.value = (2.0 * dryMass).toFixed(3);
        
        let output = "";
        output += "Target Volume: " + targetVolume.value + " mL\n";
        output += "OD600: " + od600.value + "\n";
        output += "Conversion Factor: " + conversionFactor.value + " cells/mL\n";
        output += "Total Cell Count: " + targetCellCount.value + " cells\n";
        output += "Dry Mass Conversion: " + dryMassConversion.value + " picograms/cell\n";
        output += "Total Dry Mass: " + targetDryMass.value + " g\n";
        output += "---- Medium Components per Calculated Dry Mass ----\n";
        output += "Na₂HPO₄: " + Na2HPO4.value + " g\n";
        output += "KH₂PO₄: " + KH2PO4.value + " g\n";
        output += "NaCl: " + NaCl.value + " g\n";
        output += "NH₄Cl: " + NH4Cl.value + " g\n";
        output += "MgSO₄·7H₂O: " + MgSO4.value + " g\n";
        output += "CaCl₂·2H₂O: " + CaCl2.value + " g\n";
        output += "Glucose: " + Glucose.value + " g\n";
        output += "------------------------------\n";
        appendToConsole(output);
    });
    calculateButton.click();
}

// Append the provided text to the end of the virtual console textarea
function appendToConsole(text) {
    const virtualConsole = document.getElementById('virtualConsole');
    virtualConsole.value += text;
    // Optionally scroll to the bottom if content overflows:
    virtualConsole.scrollTop = virtualConsole.scrollHeight;
}
