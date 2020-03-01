import multiprocessing
import sys
import os
from datetime import datetime
from watchdog.observers import Observer
from filemonitor import FileEventHandler
from slips.core.database import __database__
import configparser
import time
import json
import traceback

# Input Process
class InputProcess(multiprocessing.Process):
    """ A class process to run the process of the flows """
    def __init__(self, outputqueue, profilerqueue, input_type, input_information, config, packet_filter):
        multiprocessing.Process.__init__(self)
        self.outputqueue = outputqueue
        self.profilerqueue = profilerqueue
        self.config = config
        # Start the DB
        __database__.start(self.config)
        self.input_type = input_type
        self.input_information = input_information
        self.zeek_folder = './zeek_files'
        self.nfdump_output_file = 'nfdump_output.txt'
        self.nfdump_timeout = None
        self.name = 'input'
        # Read the configuration
        self.read_configuration()
        # If we were given something from command line, has preference over the configuration file
        if packet_filter:
            self.packet_filter = "'" + packet_filter + "'"
        self.event_handler = None
        self.event_observer = None

    def read_configuration(self):
        """ Read the configuration file for what we need """
        # Get the pcap filter
        try:
            self.packet_filter = self.config.get('parameters', 'pcapfilter')
        except (configparser.NoOptionError, configparser.NoSectionError, NameError):
            # There is a conf, but there is no option, or no section or no configuration file specified
            self.packet_filter = 'ip or not ip'

    def print(self, text, verbose=1, debug=0):
        """ 
        Function to use to print text using the outputqueue of slips.
        Slips then decides how, when and where to print this text by taking all the prcocesses into account

        Input
         verbose: is the minimum verbosity level required for this text to be printed
         debug: is the minimum debugging level required for this text to be printed
         text: text to print. Can include format like 'Test {}'.format('here')
        
        If not specified, the minimum verbosity level required is 1, and the minimum debugging level is 0
        """

        vd_text = str(int(verbose) * 10 + int(debug))
        self.outputqueue.put(vd_text + '|' + self.name + '|[' + self.name + '] ' + str(text))

    def read_nfdump_file(self) -> int:
        """
        A binary file generated by nfcapd can read by nfdump.
        The task for this function is watch the nfdump file and if any new line is there, read it.
        """
        file_handler = None
        next_line = None
        last_updated_file_time = datetime.now()
        lines = 0
        while True:
            if not file_handler:
                # We will open here because we do not know when nfdump will open the file.
                try:
                    file_handler = open(self.nfdump_output_file, 'r')
                except FileNotFoundError:
                    # Tryto wait for nfdump to generate output file.
                    time.sleep(1)
                    self.print('The output file for nfdump is still not created.', 0, 1)
                    continue

            if next_line is None:
                # Try to read next line from input file.
                nfdump_line = file_handler.readline()
                if nfdump_line:
                    # We have something to read.
                    # Is this line a valid line?
                    try:
                        # The first item of nfdump output is timestamp.
                        # So the first letter of timestamp should be digit.
                        ts = nfdump_line.split(',')[0]
                        if not ts[0].isdigit():
                            # The first letter is not digit -> not valid line.
                            # TODO: What is this valid line check?? explain
                            continue
                    except IndexError:
                        # There is no first item in  the line.
                        continue

                    # We have a new line.
                    last_updated_file_time = datetime.now()
                    next_line = nfdump_line
                else:
                    # There is no new line.
                    if nfdump_line is None:
                        # Verify that we didn't have any new lines in the last TIMEOUT seconds.
                        now = datetime.now()
                        diff = now - last_updated_file_time
                        diff = diff.seconds
                        if diff >= self.nfdump_timeout:
                            # Stop the reading of the file.
                            break

                    # No new line. Continue.
                    continue

            self.print("	> Sent Line: {}".format(next_line), 0, 3)
            self.profilerqueue.put(next_line)
            # print('sending new line: {}'.format(next_line))
            next_line = None
            lines += 1

        file_handler.close()
        return lines

    def read_zeek_files(self) -> int:
        # Get the zeek files in the folder now
        zeek_files = __database__.get_all_zeek_file()
        open_file_handlers = {}
        time_last_lines = {}
        cache_lines = {}
        # Try to keep track of when was the last update so we stop this reading
        last_updated_file_time = datetime.now()
        lines = 0

        while True:
            # Do...
             
            # Go to all the files generated by Zeek and read them
            # print('zeek files: {}'.format(zeek_files))
            for filename in zeek_files:
                # Update which files we know about
                try:
                    file_handler = open_file_handlers[filename]
                    # We already opened this file
                    # self.print('Old File {}'.format(filename))
                except KeyError:
                    # First time we opened this file.
                    # Ignore the files that do not contain data.
                    if 'capture_loss' in filename or 'loaded_scripts' in filename or 'packet_filter' in filename or 'stats' in filename or 'weird' in filename or 'reporter' in filename:
                        continue
                    file_handler = open(filename + '.log', 'r')
                    open_file_handlers[filename] = file_handler
                    # self.print('New File {}'.format(filename))

                # Only read the next line if the previous line was sent
                try:
                    temp = cache_lines[filename]
                    # We have still something to send, do not read the next line from this file
                except KeyError:
                    # We don't have any waiting line for this file, so proceed
                    zeek_line = file_handler.readline()
                    # self.print('Reading from file {}, the line {}'.format(filename, zeek_line))
                    # self.print('File {}, read line: {}'.format(filename, zeek_line))
                    # Did the file ended?
                    if not zeek_line:
                        # We reached the end of one of the files that we were reading. Wait for more data to come
                        continue

                    # Since we actually read something form any file, update the last time of read
                    last_updated_file_time = datetime.now()
                    try:
                        # Convert from json to dict
                        line = json.loads(zeek_line)
                        # All bro files have a field 'ts' with the timestamp.
                        # So we are safe here not checking the type of line
                        timestamp = line['ts']
                        # Add the type of file to the dict so later we know how to parse it
                        line['type'] = filename
                    except json.decoder.JSONDecodeError:
                        # It is not JSON format. It is tab format line.
                        line = zeek_line
                        # Ignore comments at the beginning of the file.
                        try:
                            if line[0] == '#':
                                continue
                        except IndexError:
                            continue
                        # Slow approach.
                        timestamp = line.split('\t')[0]
                        # Faster approach, but we do not know if
                        # line = line.rstrip()
                        # line = line + '\t' + str(filename)

                    time_last_lines[filename] = timestamp

                    # self.print('File {}. TS: {}'.format(filename, timestamp))
                    # Store the line in the cache
                    # self.print('Adding cache and time of {}'.format(filename))
                    cache_lines[filename] = line

            # Out of the for that check each Zeek file one by one
            # self.print('Out of the for.')
            # self.print('Cached lines: {}'.format(str(cache_lines)))

            # If we don't have any cached lines to send, it may mean that new lines are not arriving. Check
            if not cache_lines:
                # Verify that we didn't have any new lines in the last 10 seconds. Seems enough for any network to have ANY traffic
                now = datetime.now()
                diff = now - last_updated_file_time
                diff = diff.seconds
                if diff >= self.bro_timeout:
                    # It has been 10 seconds without any file being updated. So stop the while
                    # Get out and sto Zeek
                    break

            # Now read lines in order. The line with the smallest timestamp first
            file_sorted_time_last_lines = sorted(time_last_lines, key=time_last_lines.get)
            # self.print('Sorted times: {}'.format(str(file_sorted_time_last_lines)))
            try:
                key = file_sorted_time_last_lines[0]
            except IndexError:
                # No more sorted keys. Just loop waiting for more lines
                # It may happened that we check all the files in the folder, and there is still no file for us.
                # To cover this case, just refresh the list of files
                # self.print('Getting new files...')
                # print(cache_lines)
                zeek_files = __database__.get_all_zeek_file()
                time.sleep(1)
                continue

            # Description??
            line_to_send = cache_lines[key]
            # self.print('Line to send from file {}. {}'.format(key, line_to_send))
            # SENT
            self.print("	> Sent Line: {}".format(line_to_send), 0, 3)
            self.profilerqueue.put(line_to_send)
            # Count the read lines
            lines += 1
            # Delete this line from the cache and the time list
            # self.print('Deleting cache and time of {}'.format(key))
            del cache_lines[key]
            del time_last_lines[key]

            # Get the new list of files. Since new files may have been created by Zeek while we were processing them.
            zeek_files = __database__.get_all_zeek_file()

        # We reach here after the break produced if no zeek files are being updated.
        # No more files to read. Close the files
        for file in open_file_handlers:
            self.print('Closing file {}'.format(file), 3, 0)
            open_file_handlers[file].close()
        return lines

    def run(self):
        try:
            # Process the file that was given
            lines = 0
            if self.input_type == 'file':
                """ 
                Path to the flow input file to read. It can be a Argus binetflow flow,
                a Zeek conn.log file or a Zeek folder with all the log files. 
                """

                # If the type of file is 'file (-f) and the name of the file is '-' then read from stdin
                if not self.input_information or self.input_information == '-':
                    # By default read the stdin
                    sys.stdin.close()
                    sys.stdin = os.fdopen(0, 'r')
                    file_stream = sys.stdin
                    for line in file_stream:
                        self.print('	> Sent Line: {}'.format(line.replace('\n', '')), 0, 3)
                        self.profilerqueue.put(line)
                        lines += 1

                # If we were given a filename, manage the input from a file instead
                elif self.input_information:
                    try:
                        # Try read a file.
                        file_stream = open(self.input_information)
                        for line in file_stream:
                            self.print('	> Sent Line: {}'.format(line.replace('\n', '')), 0, 3)
                            self.profilerqueue.put(line)
                            lines += 1
                    except IsADirectoryError:
                        # Add all log files to database.
                        for file in os.listdir(self.input_information):
                            # Remove .log extension and add file name to database.
                            extension = file[-4:]
                            if extension == '.log':
                                file_name_without_extension = file[:-4]
                                __database__.add_zeek_file(self.input_information + '/' + file_name_without_extension)

                        # We want to stop bro if no new line is coming.
                        self.bro_timeout = 1
                        # time.sleep(3)
                        lines = self.read_zeek_files()


                self.profilerqueue.put("stop")
                self.outputqueue.put("01|input|[In] No more input. Stopping input process. Sent {} lines ({}).".format(lines, datetime.now().strftime('%Y-%m-%d--%H:%M:%S')))

                self.outputqueue.close()
                self.profilerqueue.close()

                return True
            # Process the binary nfdump file.
            elif self.input_type == 'nfdump':
                # Its not good to read the nfdump file to disk.
                command = 'nfdump -b -N -o csv -r ' + self.input_information + ' >  ' + self.nfdump_output_file
                os.system(command)
                self.nfdump_timeout = 10
                lines = self.read_nfdump_file()
                self.print("We read everything. No more input. Stopping input process. Sent {} lines".format(lines))
                # Delete the nfdump file
                command = "rm " + self.nfdump_output_file + "2>&1 > /dev/null &"
                os.system(command)

            # Process the pcap files
            elif self.input_type == 'pcap' or self.input_type == 'interface':
                # Create zeek_folder if does not exist.
                if not os.path.exists(self.zeek_folder):
                    os.makedirs(self.zeek_folder)
                # Now start the observer of new files. We need the observer because Zeek does not create all the files
                # at once, but when the traffic appears. That means that we need
                # some process to tell us which files to read in real time when they appear
                # Get the file eventhandler
                # We have to set event_handler and event_observer before running zeek.
                self.event_handler = FileEventHandler(self.config)
                # Create an observer
                self.event_observer = Observer()
                # Schedule the observer with the callback on the file handler
                self.event_observer.schedule(self.event_handler, self.zeek_folder, recursive=True)
                # Start the observer
                self.event_observer.start()
                # Start the observer

                # This double if is horrible but we just need to change a string
                if self.input_type == 'interface':
                    # Change the bro command
                    bro_parameter = '-i ' + self.input_information
                    # We don't want to stop bro if we read from an interface
                    self.bro_timeout = 9999999999999999
                elif self.input_type == 'pcap':
                    # We change the bro command
                    bro_parameter = '-r'
                    # Find if the pcap file name was absolute or relative
                    if self.input_information[0] == '/':
                        # If absolute, do nothing
                        bro_parameter = '-r ' + self.input_information
                    else:
                        # If relative, add ../ since we will move into a special folder
                        bro_parameter = '-r ' + '../' + self.input_information
                    # This is for stoping the input if bro does not receive any new line while reading a pcap
                    self.bro_timeout = 30

                if len(os.listdir(self.zeek_folder)) > 0:
                    # First clear the zeek folder of old .log files
                    # The rm should not be in background because we must wait until the folder is empty
                    command = "rm " + self.zeek_folder + "/*.log 2>&1 > /dev/null &"
                    os.system(command)

                # Run zeek on the pcap or interface. The redef is to have json files
                # To add later the home net: "Site::local_nets += { 1.2.3.0/24, 5.6.7.0/24 }"
                command = "cd " + self.zeek_folder + "; bro -C " + bro_parameter + " local -e 'redef LogAscii::use_json=T;' -f " + self.packet_filter + " 2>&1 > /dev/null &"
                # Run zeek.
                os.system(command)

                # Give Zeek some time to generate at least 1 file.
                time.sleep(3)

                lines = self.read_zeek_files()
                self.print("We read everything. No more input. Stopping input process. Sent {} lines".format(lines))

                # Stop the observer
                self.event_observer.stop()
                self.event_observer.join()
                return True

        except KeyboardInterrupt:
            self.outputqueue.put("04|input|[In] No more input. Stopping input process. Sent {} lines".format(lines))
            try:
                self.event_observer.stop()
                self.event_observer.join()
            except NameError:
                pass
            return True
        except Exception as inst:
            self.print("Problem with Input Process.", 0, 1)
            self.print("Stopping input process. Sent {} lines".format(lines), 0, 1)
            self.print(type(inst),0,1)
            self.print(inst.args,0,1)
            self.print(inst,0,1)
            self.event_observer.stop()
            self.event_observer.join()
            self.print(traceback.format_exc())
            sys.exit(1)