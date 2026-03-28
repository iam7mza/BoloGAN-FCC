#include <iostream>
#include <fstream>
#include <string>
#include "podio/Reader.h"
#include "edm4hep/SimCalorimeterHitCollection.h"


std::string getROOTFilePath(const char* particle, int energy) {
    /*
    Constructs the file path for a given particle type and energy.
    Parameters:
    - particle: a string indicating the particle type ('e' for electron, 'pi' for pion)
    - energy: an integer representing the energy in GeV (e.g., 10, 40 ..)
    Note: Valid energy values are 10, 20, 40, 60 for electrons and 10, 20, 40, 60, 80 for pions. (Given current files)
    Returns:
    - A string containing the full path to the corresponding ROOT file.
    */
    std::string dir;
    std::string ptype = particle;
    // fixed path for INFN clusters, change if needed

    // will work on cope of the data at /storage-hpc/bologan/alhaddad/IDEA-data/ instead of /storage-hpc/bologan/IDEA-data/
    // to keep the original data intact and avoid accidental deletion or modification. 
    if (ptype == "e") {
        dir = "/storage-hpc/bologan/alhaddad/IDEA-data/emlinearity/";
    } else if (ptype == "pi") {
        dir = "/storage-hpc/bologan/alhaddad/IDEA-data/pilinearity/";
    } else {
        std::cerr << "ERROR: unknown particle type '" << ptype << "'. Use 'e' for electron or 'pi' for pion.\n";
        return "";
    }
    return dir + "IDEA_o2_v01_phi0p5_theta0p5_" + std::to_string(energy) + "gev.root";
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "[USAGE]: " << argv[0] << " <particle type> <energy [GeV]>" << std::endl;
        return 1;
    }

    const std::string particle = argv[1];
    int energy = std::stoi(argv[2]);
    std::string outDir = "/storage-hpc/bologan/alhaddad/DataCSV/";
    std::string filePath = getROOTFilePath(particle.c_str(), energy);
    std::string prefix = outDir + (particle == "e" ? "electron" : "pion")
                       + "_" + std::to_string(energy) + "GeV_";

    auto reader = podio::makeReader(filePath);


    // All branches to extract (must all be SimCalorimeterHitCollection)
    const std::vector<std::string> branches = {
        "SCEPCal_MainEdep",
        "SCEPCal_MainCcounts",
        "SCEPCal_MainScounts",
        "DRBTCher",
        "DRBTScin",
        "DRETCherLeft",
        "DRETCherRight",
        "DRETScinLeft",
        "DRETScinRight"
    };

    // Open one file per branch
    std::map<std::string, std::ofstream> files;
    for (const auto& branch : branches) {
        std::string fname = prefix + branch + ".csv";
        files[branch].open(fname);
        if (!files[branch].is_open()) {
            std::cerr << "[ERROR] Cannot open: " << fname << std::endl;
            return 1;
        }
        files[branch] << "Event,Hit,Energy,PosX,PosY,PosZ,cellID\n";
        std::cout << "[INFO] Opened: " << fname << std::endl;
    }

    // Single pass over events — write all branches at once
    size_t nEvents = reader.getEvents();
    for (size_t i = 0; i < nEvents; ++i) {
        auto event = reader.readNextEvent();

        for (const auto& branch : branches) {
            auto& hits = event.get<edm4hep::SimCalorimeterHitCollection>(branch);
            int j = 1;
            for (const auto& hit : hits) {
                files[branch] << i << ","
                              << j++ << ","
                              << hit.getEnergy() << ","
                              << hit.getPosition().x << ","
                              << hit.getPosition().y << ","
                              << hit.getPosition().z << ","
                              << hit.getCellID() << "\n";
            }
        }

        if (i % 500 == 0)
            std::cout << "[INFO] Processed event " << i << "/" << nEvents << std::endl;
    }

    // Close all files
    for (const auto& branch : branches) files[branch].close();

    std::cout << "[INFO] Done." << std::endl;
    return 0;
}