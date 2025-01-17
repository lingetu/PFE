//#include "../../mysql.h"
#include <iostream>
#include <string.h>
#include "Bdd.h"
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/flann/general.h>
//#include <cppconn/driver.h>
//#include <cppconn/exception.h>
//#include <cppconn/resultset.h>
//#include <cppconn/statement.h>

using namespace std;

//Bdd::Bdd() {
//}

void dbCommunication(int nbCollisions, float stress, int level, int gameTime)
{

    MYSQL sock;
    MYSQL_ROW row = NULL;
    MYSQL_RES* res = NULL;

    //mysql_init(&sock);
    MYSQL *conn = mysql_init(0);
    const char* host = "localhost";
    const char* user = "root";
    const char* pass = "root";
    const char* db = "sqllabyar";
    const int port = NULL;

   


    //const char* host = "mysql.pedaweb.univ-amu.fr";
    //const char* user = "f19023768";
    //const char* pass = "yCklVTJKEqsZUyT";
    //const char* db = "f19023768";
    //const int port = NULL;
    cout << "yes" << endl;
    conn = mysql_real_connect(conn, host, user, pass, db, port, NULL, 0);
    cout << "fini" << endl;
    if (conn)
        cout << "Connexion a la base reussie !" << endl;
    else
        cout << "Connexion impossible." << endl;
    std::string str;

    try {
        ///auto* stmt = mysql_stmt_init(conn);
        ///std::string query = "Insert INTO `stat_joueur` (`idJoueur`, `nb_collisions`, `stress`, `niveau`, `temps`) VALUES (NULL, `?`, `?`, `?`, `?`); ";
        ///mysql_stmt_prepare(stmt, query.c_str(), query.length());
        ///MYSQL_BIND bind[4];
        ///
        ///memset(bind, 0, sizeof(bind));
        ///bind[0].buffer_type = MYSQL_TYPE_INT24;
        ///bind[0].buffer = &nbCollisions;
        ///bind[0].is_null = 0;
        ///bind[0].length = 0;
        ///
        ///
        ///memset(bind, 0, sizeof(bind));
        ///bind[1].buffer_type = MYSQL_TYPE_FLOAT;
        ///bind[1].buffer = &stress;
        ///bind[1].is_null = 0;
        ///bind[1].length = 0;
        ///
        ///memset(bind, 0, sizeof(bind));
        ///bind[2].buffer_type = MYSQL_TYPE_INT24;
        ///bind[2].buffer = &level;
        ///bind[2].is_null = 0;
        ///bind[2].length = 0;
        ///
        ///
        ///memset(bind, 0, sizeof(bind));
        ///bind[3].buffer_type = MYSQL_TYPE_INT24;
        ///bind[3].buffer = &gameTime;
        ///bind[3].is_null = 0;
        ///bind[3].length = 0;
        ///
        ///mysql_stmt_bind_param(stmt, bind);
        ///mysql_stmt_execute(stmt);
        ///mysql_stmt_close(stmt);
        
        //sql::PreparedStatement* prep_stmt;
        //const char* tableName = "foo";
        //PreparedStatement* prep_stmt;
        //prep_stmt = conn->prepareStatement("SELECT * FROM ?");
        //prep_stmt->setString(1, tableName);
        //prep_stmt->execute();
    }
    catch (int e) {
        printf("oskour"+e);
    }




    //memset(bind, 0, sizeof(bind));

    //str = "Insert INTO `stat_joueur` (`idJoueur`, `nb_collisions`, `temps`, `stress`, `niveau`) VALUES (NULL, '"+ nbCollisions +"', '" + gameTime + "', '" + stress + "', '" + level + "')";
    str = "Insert INTO `stat_joueur` (`idJoueur`, `nb_collisions`, `temps`, `stress`, `niveau`) VALUES (NULL, '";
    str += to_string(nbCollisions);  
    str += "', '";
    str += to_string(gameTime);
    str += "', '";
    str += to_string(stress);
    str += "', '";
    str += to_string(level);
    str += "')";
    cout << str << endl;
    //str = "Insert INTO `stat_joueur` (`idJoueur`, `nb_collisions`, `temps`, `stress`, `niveau`) VALUES(NULL, '32', '855', '30.000000', '4')";

    mysql_query(conn, str.c_str());
    //res = mysql_use_result(conn);
    ///mysql_query(&sock, str.c_str());
    ///res = mysql_use_result(&sock);
    //row = mysql_fetch_row(res);

    //if (row) {
    //    cout << "Ligne correctement ajoutée"<< row << endl;
    //}
    //else {
    //    cout << "Problème durant l'insertion dans la BDD" << endl;
    //}
    //return;
}


int getnextID() {
    MYSQL sock;
    MYSQL_ROW row = NULL;
    MYSQL_RES* res = NULL;

    //mysql_init(&sock);
    MYSQL* conn = mysql_init(0);
    const char* host = "localhost";
    const char* user = "root";
    const char* pass = "root";
    const char* db = "sqllabyar";
    const int port = NULL;


    // Connexion au serveur MySQL (sans base de données pour commencer)
    conn = mysql_real_connect(conn, host, user, pass, NULL, port, NULL, 0);

    if (conn) {
        cout << "Connexion au serveur MySQL réussie !" << endl;

        /* INSERER CODE DB ICI */
        // Création de la base de données
        const char* createDbQuery = "CREATE DATABASE IF NOT EXISTS sqllabyar;";
        if (mysql_query(conn, createDbQuery) == 0) {
            cout << "Base de données 'sqllabyar' créée ou déjà existante." << endl;

            // Sélection de la base de données
            if (mysql_select_db(conn, db) == 0) {
                cout << "Base de données 'sqllabyar' sélectionnée avec succès." << endl;

                // Création de la table
                const char* createTableQuery =
                    "CREATE TABLE IF NOT EXISTS stat_joueur ("
                    "idJoueur INT AUTO_INCREMENT PRIMARY KEY, "
                    "nb_collisions INT NOT NULL, "
                    "temps INT NOT NULL, "
                    "stress FLOAT NOT NULL, "
                    "niveau INT NOT NULL);";

                if (mysql_query(conn, createTableQuery) == 0) {
                    cout << "Table 'stat_joueur' créée ou déjà existante." << endl;
                }
                else {
                    cerr << "Erreur lors de la création de la table : " << mysql_error(conn) << endl;
                }
            }
            else {
                cerr << "Erreur lors de la sélection de la base de données : " << mysql_error(conn) << endl;
            }
        }
        else {
            cerr << "Erreur lors de la création de la base de données : " << mysql_error(conn) << endl;
        }
    }
    else {
        cerr << "Connexion au serveur MySQL impossible : " << mysql_error(conn) << endl;
    }

    //const char* host = "mysql.pedaweb.univ-amu.fr";
    //const char* user = "f19023768";
    //const char* pass = "yCklVTJKEqsZUyT";
    //const char* db = "f19023768";
    //const int port = NULL;
    cout << "yes" << endl;
    
    cout << "fini" << endl;
    if (conn)
        cout << "Connexion a la base reussie !" << endl;
    else
        cout << "Connexion impossible." << endl;
    std::string str;

    str = "SELECT MAX(`idJoueur`) FROM `stat_joueur` WHERE 1";

    mysql_query(conn, str.c_str());

    res = mysql_store_result(conn);
    int num_fields = mysql_num_fields(res);
    // Fetch all rows from the result
    while ((row = mysql_fetch_row(res)))
    {
        // Print all columns
        for (int i = 0; i < num_fields; i++)
        {
            // Make sure row[i] is valid!
            if (row[i] != NULL) {
                int val;
                sscanf(row[i], "%d", &val);
                cout << row[i]<<", "<< val << endl;
                return val+1;
            }
            else
                cout << "NULL" << endl;
        }
    }
    return 1;
}

