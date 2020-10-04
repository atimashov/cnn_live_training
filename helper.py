from psycopg2 import connect
from datetime import datetime
import pandas as pd

SCHEMA = 'cnn_live_training'
def get_con():
    return connect(
        database = 'dl_playground', user = 'postgres', password = 'qmzpqmzp', host = 'localhost', port = 5432
    )

def update_db(params, dt_start, epoch, step, num_samples, steps, tloss, tacc, vloss, vacc, maps):
    with get_con() as con:
        cur = con.cursor()
        # statistics
        cols = '(dt_started, model_name, epoch, step, optimizer, learning_rate, weight_decay, dropout, dt, train_loss, train_accuracy, validate_loss, validate_accuracy)'
        prev_lr, prev_wd, prev_do, prev_opt, _ = params
        q_ins = """
                INSERT INTO {}.statistics {}
                VALUES ('{}', '{}', {}, {}, '{}', {}, {}, {}, '{}', {}, {}, {}, {})
                """.format(
            SCHEMA, cols, dt_start, 'MyAlexNet', epoch, step, prev_opt, prev_lr, prev_wd, prev_do, datetime.now(),
            round(tloss / steps, 4), round(100 * float(tacc) / num_samples, 2), round(vloss / steps, 4),
            round(vacc / steps, 2)
        )
        cur.execute(q_ins)

        # activation maps
        q_del = "DELETE FROM {}.activations".format(SCHEMA)
        cur.execute(q_del)
        cols = '(nn_part, layer_type, number, dt, weights, num_weights)'
        q_ins = """
    		INSERT INTO {}.activations {}
    		VALUES {}
    		""".format(SCHEMA, cols, ', '.join(maps))
        cur.execute(q_ins)
        con.commit()

def get_params(start = False):
    query = """
		SELECT *
		FROM {}.parameters
	""".format(SCHEMA)
    with get_con() as conn:
        df = pd.read_sql(query, conn)
    if start and (df.shape[0] == 0 or (datetime.now() - df['dt_updated'][0]).seconds >= 30):
        return None
    cols = ['learning_rate', 'weight_decay', 'dropout', 'optimizer', 'stop_train']
    return df[cols].values[0]

def update_params(opt, lr, wd, do, dt_start):
    with get_con() as con:
        q_upd = """
            UPDATE {}.parameters
            SET optimizer = '{}',
                learning_rate = {},
                weight_decay = {},
                dropout = {},
                dt_updated = '{}',
                stop_train = False
        """.format(SCHEMA, opt, lr, wd, do, dt_start)
        cur = con.cursor()
        cur.execute(q_upd)
        con.commit()

def stop_train():
   with get_con() as con:
        cur = con.cursor()
        q_upd = """
            UPDATE {}.parameters
            SET stop_train = True
        """.format(SCHEMA)
        cur.execute(q_upd)
        con.commit()
