// quick inspection of .root files
// Usage:
// root -l -q 'inspectROOT.cpp("e", 10, 3)"
// args : 1- "e" or "pi" for either electron or pion
// 2 - energy (optional)
// 3 - event (optional)


void inspectROOT(const char* particle = "e", int energy = 10, int eventToShow = 0) {
	
	// dir: NOTE this is meant to run on INFN clusters.. and the directory must be changed otherwise
	std::string dir;
	std::string ptype = particle;
	if (ptype == "e"){dir = "/storage-hpc/bologan/IDEA-data/emlinearity/";}
	else if (ptype == "pi"){dir = "/storage-hpc/bologan/IDEA-data/pilinearity/";} 
	else {std::cerr << "ERROR: unknow particle type '" << ptype << "'. Use 'e' for electron or 'pi' for pion.\n";
		return;}
	
	std::cout <<"\ndir = " << dir<<'\n';
	
	// filename
	std::string filename = dir + "IDEA_o2_v01_phi0p5_theta0p5_" + std::to_string(energy) + "gev.root";

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
