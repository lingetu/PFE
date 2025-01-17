#pragma once
#include <mysql.h>
//class Bdd {
//private:
//	int stress;
//	int nbCollisions;
//	int level;
//	int gameTime;
//	int pseudo;
//
//public:
//	//Bdd();
//};

//void insertData();
void dbCommunication(int nbCollisions, float stress, int level, int gameTime);
int getnextID();