#include <iostream>
#include <fstream>
#include "rootFunction.h"
#include <TFile.h>
#include <edm4hep/SimCalorimeterHitData.h>

int main(int argc, char* argv[]) {
    /*
    Usage: ./extractFromROOT <particle type> <energy [GeV]>
    Example: ./extractFromROOT e 10; or ./extractFromROOT pi 40
    This program extracts the calorimeter hit data for both SCEPCal_MainEdep and other branches such as the one 
    for cherenkov hits. 
    The exact branches are tbd. 
    */
    if (argc != 3) {
        std::cerr << "[USAGE]: " << argv[0] << " <particle type> <energy [GeV]>" << std::endl;
        return 1;
    }

    const char* particle = argv[1];
    int energy = std::stoi(argv[2]);

    std::string filePath = getROOTFilePath(particle, energy);
    std::cout << "[INFO] Opening ROOT file: " << filePath << std::endl;
    TFile* file = TFile::Open(filePath.c_str());
    if (!file || file->IsZombie()) {
        std::cerr << "[ERROR] Error opening file: " << filePath << std::endl;
        return 1;
    }
    TTree* tree = (TTree*)file->Get("events");
    if (!tree) {
        std::cerr << "[ERROR] TTree 'tree' not found in file: " << filePath << std::endl;
        file->Close();
        return 1;
    }

    // Getting SCEPcal_MainEdep content happens HERE.
    auto hits= getBranchContent<std::vector<edm4hep::SimCalorimeterHitData>>(tree, "SCEPCal_MainEdep");

    // number of hits to be extracted
    std::cout << "[INFO] Number of events: " << hits.size() << std::endl;

    size_t totalHits = 0;
    for (const auto& hit : hits) {
        totalHits += hit.size();
    }
    std::cout << "[INFO] Total number of hits: " << totalHits << std::endl;

    // creating CSV file
    std::string outDir = "/storage-hpc/bologan/alhaddad/CSVoutput/"; //CAFEUL THE PATH
    std::string csvFileName;

    if (particle == std::string("e")) {
        csvFileName = outDir + "electron_" + std::to_string(energy) + "GeV_SCEPCal_MainEdep.csv";
    } else if (particle == std::string("pi")) {
        csvFileName = outDir + "pion_" + std::to_string(energy) + "GeV_SCEPCal_MainEdep.csv";
    }

    std::ofstream csvFile(csvFileName);
    if (!csvFile.is_open()) {
        std::cerr << "[ERROR] Error creating CSV file: " << csvFileName << std::endl;
        file->Close();
        return 1;
    }
    
    std::cout << "[INFO] Extracting data and writing to CSV file: " << csvFileName << std::endl;
    // Extraction
    csvFile << "Event,Hit,Energy,PosX,PosY,PosZ,cellID\n";
    for (int i = 0; i < hits.size(); ++i) { // i is the event number
        for (int j = 0; j < hits[i].size(); ++j) { // j is the hit number in event i
            const auto& hit = hits[i][j];
            csvFile << i << ","
                    << j + 1 << "," // hit number starts from 1 for later count checking
                    << hit.energy << ","
                    << hit.position.x << ","
                    << hit.position.y << ","
                    << hit.position.z << ","
                    << hit.cellID << "\n";
        }
    }
    std::cout << "[INFO] Data extraction and CSV writing completed successfully." << std::endl;
    csvFile.close();

    // Memory cleanup
    hits.clear();
    hits.shrink_to_fit();

    // ======================================================================================
    // TODO: extracting the other branches happens here
    // NOTE: extract into new files
    // ======================================================================================


    
    // FIN
    file->Close();
    return 0;
}
