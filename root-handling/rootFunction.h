// functions to handle root files because ROOT has horrible documentation
// NOTE TO SELF: only open .root files in READ mode!!


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

// Extracting branch names from TTree and returning as a vector of strings
std::vector<std::string> getBranchNames(TTree* tree) {
    /*
    Extracts names of all branches from a given TTree and returns them as a vector of strings.
    Parameters:
    - tree: a pointer to the TTree from which to extract branch names.
    Returns:
    - A vector of strings, each containing the name of a branch in the TTree.
    */
    std::vector<std::string> branchNames;
    TObjArray* branches = tree->GetListOfBranches();
    for (int i = 0; i < branches->GetEntries(); ++i) {
        TBranch* branch = (TBranch*)branches->At(i);
        branchNames.push_back(branch->GetName());
    }
    return branchNames;
}

// Extract Calorimeter hits (Careful the types) 


// some assumptions!! NOTW: please verify
// energy deposite in calorimeter is assumed to be stored in SCEPCal_MainEdep
// type vector<edm4hep::SimCalorimeterHitData>

// NOTE: this is the template for extracting data from branches
std::vector<edm4hep::SimCalorimeterHitData> getCalorimeterHits(TTree* tree, int entry) {
    /*
    Extracts calorimeter hit data from a specified entry in a TTree.
    Parameters:
    - tree: a pointer to the TTree containing the calorimeter hit data.
    - entry: the entry number from which to extract the data.
    Returns:
    - A vector of edm4hep::SimCalorimeterHitData objects representing the calorimeter hits for the specified entry.
    Note: Energy deposite is in hits.energy
    */
    std::vector<edm4hep::SimCalorimeterHitData> hits;
    std::vector<edm4hep::SimCalorimeterHitData>* hitPtr = nullptr; // code breaks if I try to directly set branch address to hits
    tree->SetBranchAddress("SCEPCal_MainEdep", &hitPtr);
    tree->GetEntry(entry);
    if (hitPtr) {
        hits = *hitPtr; // copy data to return vector
    }
    return hits;    

    // note this is very inefficient as it gets only one entry at a time. 
    // the main motivation behind this one is to get familiar with the process of extracting data from branches.
    // TODO: optimize
}