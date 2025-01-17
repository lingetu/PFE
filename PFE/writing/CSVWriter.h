#include <string>
#include <fstream>
#include <vector>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream
#include <iostream>
#include <ctime>

class CSVWriter {
private :
	std::string filename;
	
	int lastUserID;
	std::string date;
	int gameTime;
	int collisionNumber;
	float stressAverage;

public:
	CSVWriter(std::string filename, int lastUserID, int gameTime, int collisionNumber, float stressAverage);
	void writeColumnsName();
	bool doesFileExists();
	int getLastUserID();
	void writeLine();




};