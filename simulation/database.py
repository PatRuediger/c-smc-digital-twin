import sqlite3
import os


class DatabaseManager:
    """Manages all database operations for simulation results."""

    def __init__(self, db_path):
        """
        Initializes the manager with the path to the database location.

        :param db_path: The folder path from config.json.
        """
        filename = "stripGen_results.db"
        self.db_path = os.path.join(db_path, filename)
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        except OSError as e:
            print(f"Error creating folder for database: {e}")

        self.conn = None
        print(f"Database will be saved at: {self.db_path}")

    def connect(self):
        """Establishes the connection to the database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"Successfully connected to database '{self.db_path}'.")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            self.conn = None

    def close(self):
        """Closes the database connection if open."""
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    def create_table(self):
        """Creates the 'results' table if it does not already exist."""
        if not self.conn:
            print("No database connection. Cannot create table.")
            return

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS results (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

            -- Configuration parameters
            number_of_strips INTEGER,
            spawn_interval INTEGER,
            simulation_frames INTEGER,
            outlet_y_range TEXT,
            outlet_x_pos REAL,
            outlet_z_pos REAL,
            belt_speed REAL,
            base_plate_dims TEXT,
            belt_start_x REAL,
            belt_stop_delay_frames INTEGER,
            spawn_clearance_radius REAL,
            spawn_height_offset REAL,
            storage_x_offset_factor REAL,
            storage_z_offset_factor REAL,
            measurement_frame INTEGER,
            measurement_box_center TEXT,
            measurement_box_dims TEXT,
            random_seed INTEGER,

            -- Result values
            density_3d REAL,
            strips_inside_box_count INTEGER,
            density_2d REAL,

            -- Placeholder for future data
            image_path_aolp TEXT,
            bbox_path TEXT
        );
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(create_table_sql)
            print("Table 'results' successfully created or already exists.")
        except sqlite3.Error as e:
            print(f"Error creating table: {e}")

    def insert_result(self, config, density_3d, density_2d, strips_inside_box_count,
                      measurement_frame, image_path_aolp=None, bbox_path=None):
        """
        Inserts a new record with configuration and result values into the table.

        :param config: The SimulationConfig object with all parameters.
        :param density_3d: The calculated 3D density value.
        :param density_2d: The calculated 2D density value.
        :param strips_inside_box_count: Number of strips inside the measurement box.
        :param measurement_frame: The frame number that was analyzed.
        :param image_path_aolp: (Optional) The path to the AoLP image.
        :param bbox_path: (Optional) The path to the YOLO annotation file.
        """
        if not self.conn:
            print("No database connection. Cannot insert data.")
            return

        insert_sql = """
        INSERT INTO results (
            number_of_strips, spawn_interval, simulation_frames, outlet_y_range,
            outlet_x_pos, outlet_z_pos, belt_speed, base_plate_dims, belt_start_x,
            belt_stop_delay_frames, spawn_clearance_radius, spawn_height_offset,
            storage_x_offset_factor, storage_z_offset_factor, measurement_frame,
            measurement_box_center, measurement_box_dims, random_seed,
            density_3d, density_2d, strips_inside_box_count, image_path_aolp, bbox_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        data_tuple = (
            config.number_of_strips, config.spawn_interval, config.simulation_frames,
            str(config.outlet_y_range), config.outlet_x_pos, config.outlet_z_pos,
            config.belt_speed, str(config.base_plate_dims), config.belt_start_x,
            config.belt_stop_delay_frames, config.spawn_clearance_radius,
            config.spawn_height_offset, config.storage_x_offset_factor,
            config.storage_z_offset_factor, measurement_frame,
            str(config.measurement_box_center), str(config.measurement_box_dims),
            config.random_seed, density_3d, density_2d, strips_inside_box_count,
            image_path_aolp, bbox_path
        )
        try:
            cursor = self.conn.cursor()
            cursor.execute(insert_sql, data_tuple)
            self.conn.commit()
            print(f"New run (ID: {cursor.lastrowid}) successfully written to the database.")
        except sqlite3.Error as e:
            print(f"Error inserting data: {e}")
