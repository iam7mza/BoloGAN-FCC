// functions to handle root files because ROOT has horrible documentation
// NOTE TO SELF: only open .root files in READ mode!!

#include <string>
#include <vector>
#include <iostream>
#include <TTree.h>
#include <TBranch.h>
#include <TObjArray.h>

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

    WANRING: ram handeling; this function loads all the dataset into a single vector.. if the dataset is too large, it might cause memory issues. 
    TODO: split the loading into batches (Depending on available memory).
    */
    std::vector<std::string> branchNames;
    TObjArray* branches = tree->GetListOfBranches();
    for (int i = 0; i < branches->GetEntries(); ++i) {
        TBranch* branch = (TBranch*)branches->At(i);
        branchNames.push_back(branch->GetName());
    }
    return branchNames;
}

// Extracting content from a branch.
template<typename T>
std::vector<T> getBranchContent(TTree* tree, const std::string& branchName, int entry = -1) {
    /*
    Extracts content from a specified branch in a TTree for a given entry or all entries
    Parameters:
    - tree: a pointer to the TTree containing the branch
    - branchName: the name of the branch from which to extract data
    - entry: the entry number to extract (default -1 for all entries)
    Returns:
    - A vector of type T containing the data from the specified branch and entry/entries.
    Note: if getting a single entry, the returned value is still a vector and therefore data must be extracted with `result[0]`
    IMPORTANT: type must be specified before calling the function eg `getBranchContent<std::vector<edm4hep::SimCalorimeterHitData>>(t, "SCEPCal_MainEdep", 4999);`

    Accessing the data: 
    1- A given entry can be accessed with `result[nEntry]` as the return is a vector (note: for one entry nEntry is 0)
    2- A sigle event/entry might consist of multiple hits. To access indivisual hits, a for loop over the results[nEntry] is needed. eg `for(const auto* hit : results[nEntry]) {...}`
    3- A given hit might have multiple subfield. eg: `hit.energy`, `hit.position.x` .. etc
    */

    T* dataPtr = nullptr;
    tree->SetBranchAddress(branchName.c_str(), &dataPtr); 
    int nEntries = tree->GetEntries();

    if (entry >= nEntries) {
        std::cerr << "ERROR: could not get entry " << entry << ". Entry number must not exceed " << nEntries - 1 << ".\n";
        return {};
    }

    if (entry >= 0) {
        tree->GetEntry(entry);
        if (dataPtr) return {*dataPtr};
        return {};  // explicit return, no fall-through
    }

    // entry == -1: all entries
    std::vector<T> all;
    for (int i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        if (dataPtr) all.push_back(*dataPtr);
    }
    return all;
}