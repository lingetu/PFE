#include "CSVWriter.h"

#include <string>
#include <fstream>
#include <vector>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream
#include <iostream>
#include <ctime>


CSVWriter::CSVWriter(std::string filename, int lastUserID, int gameTime, int collisionNumber, float stressAverage) {
    this->lastUserID = lastUserID;
    this->filename = filename;
    this->gameTime = gameTime;
    this->collisionNumber = collisionNumber;
    this->stressAverage = stressAverage;
}

//Fonction de vérification de l'existence du fichier de stats
bool CSVWriter::doesFileExists()
{
    std::ifstream infile(this->filename);
    return infile.good();
}

//Fonction d'écriture des premières lignes du fichier
void CSVWriter::writeColumnsName()
{
    std::ofstream myfile;
    myfile.open(this->filename);
    myfile << "userID;date;time;collisionNumber;stressAverage\n";
    myfile << "0;0;0;0;0";
    myfile.close();
}

//Ecriture d'une nouvelle ligne
void CSVWriter::writeLine() {

    std::ofstream out;
    out.open(this->filename, std::ios::app);

    //userID
    //int newUserID = getLastUserID() + 1;

    //date
    std::time_t t = std::time(0);   // get time now
    std::tm* now = std::localtime(&t);
    std::string date = std::to_string(now->tm_year + 1900) + "/" + std::to_string(now->tm_mon + 1) + "/" + std::to_string(now->tm_mday) + "/" + std::to_string(now->tm_hour) + ":" + std::to_string(now->tm_min);

    //Ecriture
    std::string str = "\n" + std::to_string(this->lastUserID) + ";" + date + ";" + std::to_string(this->gameTime) + ";" + std::to_string(this->collisionNumber) + ";" + std::to_string(this->stressAverage);
    out << str;
}


int CSVWriter::getLastUserID()
{
    std::string filename = this->filename;
    std::ifstream fin;
    int val;

    fin.open(filename);
    if (fin.is_open()) {
        fin.seekg(-1, std::ios_base::end);                // go to one spot before the EOF

        bool keepLooping = true;
        while (keepLooping) {
            char ch;
            fin.get(ch);                            // Get current byte's data

            if ((int)fin.tellg() <= 1) {             // If the data was at or before the 0th byte
                fin.seekg(0);                       // The first line is the last line
                keepLooping = false;                // So stop there
            }
            else if (ch == '\n') {                   // If the data was a newline
                keepLooping = false;                // Stop at the current position.
            }
            else {                                  // If the data was neither a newline nor at the 0 byte
                fin.seekg(-2, std::ios_base::cur);        // Move to the front of that data, then to the front of the data before it
            }
        }

        std::string lastLine;
        std::getline(fin, lastLine);                      // Read the current line

        std::stringstream ss(lastLine);

        // Get first integer which is userID
        ss >> val;

        fin.close();
    }
    return val;
}



