// quick inspection of .root files
// Usage:
// root -l -q 'inspectROOT.cpp("e", 10, 3)"
// args : 1- "e" or "pi" for either electron or pion
// 2 - energy (optional)
// 3 - event (optional)


// including functions from rootFunction.h
#include "rootFunction.h"

void inspectROOT(const char* particle = "e", int energy = 10, int eventToShow = 0) {
	

	std::string filename = getROOTFilePath(particle, energy);
	std::cout << "Opening: " << filename << "\n";
	
	// opening file
	TFile* f = TFile::Open(filename.c_str(), "READ");
	if (!f || f->IsZombie()){
		std::cerr << "ERORR: Cannot open " <<filename<<'\n';
		return;
	}

	std::cout << "\n============================\n";
	std::cout << "======= Tree Branches ======\n";
	f->ls();


	// inspecting events tree
	TTree* t = (TTree*)f->Get("events");
  	if (!t) {
        std::cerr << "ERROR: 'events' tree not found\n";
        f->Close();
        return;
   	}

	std::cout << "\n============================\n";
        std::cout << "======= Event Branches ======\n";
	t->Print();

	std::cout << "\n============================\n";
        std::cout << "======= Event Content =======\n";

	// MCParticle.spin branch is problematic and results in error
	t->SetBranchStatus("MCParticles.spin*", 0);  // disable spin branches
	t->Show(eventToShow);
	// t->SetBranchStatus("MCParticles.spin*", 1);  // to enable for the future
	f->Close();
	return;

}
