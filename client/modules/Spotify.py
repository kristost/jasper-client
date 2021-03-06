# -*- coding: utf-8-*-
import re
import logging
import difflib
import mpd as mpd2
from client.mic import Mic
#import MPDWrapper2
import time
import random

# Standard module stuff
WORDS = ["SPOTIFY"]


def handle(text, mic, profile):
    """
    Spotify module to play songs, playlists

    Arguments:
        text -- user-input, typically transcribed speech
        mic -- used to interact with the user (for both input and output)
        profile -- contains information related to the user (e.g., phone
                   number)
    """
    logger = logging.getLogger(__name__)

    kwargs = {}
    if 'mpdclient' in profile:
        if 'server' in profile['mpdclient']:
            kwargs['server'] = profile['mpdclient']['server']
        if 'port' in profile['mpdclient']:
            kwargs['port'] = int(profile['mpdclient']['port'])

    logger.debug("Preparing to start Spotify module")
    try:
        mpdwrapper = MPDWrapper(**kwargs)
    except:
        logger.error("Couldn't connect to MPD server", exc_info=True)
        mic.say("I'm sorry. It seems that Spotify is not enabled. Please " +
                "read the documentation to learn how to configure Spotify.")
        return

    mic.say("Starting Spotify...")

    # FIXME: Make this configurable
    persona = 'JASPER'

    logger.debug("Starting spotify mode")
    music_mode = MusicMode(persona, mic, mpdwrapper)
    music_mode.handleForever()
    logger.debug("Exiting spotify mode")

    return


def isValid(text):
    """
        Returns True if the input is related to Spotify.

        Arguments:
        text -- user-input, typically transcribed speech
    """
    return any(word in text.upper() for word in WORDS)


# The interesting part
class MusicMode(object):

    def __init__(self, PERSONA, mic, mpdwrapper):
        self._logger = logging.getLogger(__name__)
        self.persona = PERSONA
        # self.mic - we're actually going to ignore the mic they passed in
        self.mic = mic
        self.music = mpdwrapper

        # index spotify playlists into new dictionary and language models
        phrases = ["STOP", "CLOSE", "PLAY", "PAUSE", "NEXT", "PREVIOUS",
                   "LOUDER", "SOFTER", "LOWER", "HIGHER", "VOLUME",
                   "PLAYLIST"]
        #phrases.extend(self.music.get_soup_playlist())

        #music_stt_engine = mic.active_stt_engine.get_instance('music', phrases)

        #self.mic = Mic(mic.speaker,
        #               mic.passive_stt_engine,
        #               music_stt_engine)
        self.rickrollin = False

    def delegateInput(self, input):

        command = input.upper()

        if self.rickrollin and "PLEASE" not in command:
            random_phrase = random.choice([
                "Pardon?", 
                "I'm sorry, could you try saying that again?", 
                "I'm sorry, I don't speak 'rude'.", 
                "What's the magic word?", 
                "You didn't say 'please'."])
            self.mic.say(random_phrase)
            self.music.play()
            return
        elif self.rickrollin and "PLEASE" in command:
            self._logger.debug('Keyword "Please" detected in phrase.')
            self.mic.say("OK, since you asked so nicely.")
            self.rickrollin = False
    
        # check if input is meant to start the music module
        
        # Rickroll the user (intended for a laugh/get a rise out of them -- for emotion detection)
        if any(ext in command for ext in ["FAVORITE", "FAVOURITE"]):
            success, result = self.music.rickroll()
            if success:
                self.music.play()
                self.music.seekcur(44) # start playing at the chorus!
                self.rickrollin = True
            return
        elif "SPOTIFY" in command:
            success, result = self.music.searchadd(command)
            self._logger.info("Search success: {}; Found: {}".format(success, result))
            if success:
                self.mic.say("Playing %s" % self.music.current_song())
                self.music.play()
            else:
                self.mic.say("Sorry, I couldn't find {} by {}".format(result[0], result[1]))
                self.music.play()
            
            return
        elif "PLAYLIST" in command:
            command = command.replace("PLAYLIST", "")
        elif "PLAY LIST" in command:
            command = command.replace("PLAY LIST", "")
        elif "STOP" in command:
            #self.mic.say("Stopping music")
            self.music.stop()
            return
        elif "NEXT" in command:
            #self.mic.say("Next song")
            self.music.next()
            time.sleep(0.25)
            self.mic.say("Playing %s" % self.music.current_song())
            self.music.play()  # backwards necessary to get mopidy to work
            return
        elif "PREVIOUS" in command:
            #self.mic.say("Previous song")
            self.music.previous()
            time.sleep(0.25)
            self.mic.say("Playing %s" % self.music.current_song())
            self.music.play()  # backwards necessary to get mopidy to work
            return
        elif "CURRENT" in command:
            self.mic.say("Playing %s" % self.music.current_song())
            self.music.play()  # backwards necessary to get mopidy to work
            return
        elif "PLAY" in command:
            self.mic.say("Playing %s" % self.music.current_song())
            self.music.play()
            return
        elif "PAUSE" in command:
            #self.mic.say("Pausing music")
            #self.music.pause()
            # do nothing because listening for the keyword will pause the music anyways
            return
        elif any(ext in command for ext in ["LOUDER", "HIGHER", "TURN IT UP", "VOLUME UP"]):
            #self.mic.say("Louder")
            self.music.volume(interval=10)
            self.music.play()
            return
        elif any(ext in command for ext in ["SOFTER", "LOWER", "TURN IT DOWN", "VOLUME DOWN"]):
            #self.mic.say("Softer")
            self.music.volume(interval=-10)
            self.music.play()
            return
        elif "VOLUME" in command:
            p = re.compile('set(?: the)? volume(?:[ a-zA-Z]+)? (\d)', re.IGNORECASE)
            match = p.match(command)
            if match:
                level = int(match.group(1)) * 10                
                self.music.volume(level=level)
                self.music.play()
            return
        
        # SONG SELECTION... requires long-loading dictionary and language model
        # songs = self.music.fuzzy_songs(query = command.replace("PLAY", ""))
        # if songs:
        #     self.mic.say("Found songs")
        #     self.music.play(songs = songs)

        #     print("SONG RESULTS")
        #     print("============")
        #     for song in songs:
        #         print("Song: %s Artist: %s" % (song.title, song.artist))

        #     self.mic.say("Playing %s" % self.music.current_song())

        # else:
        #     self.mic.say("No songs found. Resuming current song.")
        #     self.music.play()

        #==========================
        # PLAYLIST SELECTION
        #playlists = self.music.fuzzy_playlists(query=command)
        #if playlists:
        #    self.mic.say("Loading playlist %s" % playlists[0])
        #    self.music.play(playlist_name=playlists[0])
        #    self.mic.say("Playing %s" % self.music.current_song())
        #else:
        #    self.mic.say("No playlists found. Resuming current song.")
        #    self.music.play()

        return False # we couldn't handle the request

    def handleForever(self):

        #self.music.play()
        #self.mic.say("Playing %s" % self.music.current_song())

        while True:

            threshold, transcribed = self.mic.passiveListen(self.persona)

            if not transcribed or not threshold:
                self._logger.info("Nothing has been said or transcribed.")
                continue

            if self.music.state() != 'pause':
                self.music.pause()

            input = self.mic.activeListen(MUSIC=True)

            if input:
                if "close" in input.lower():
                    self.mic.say("Closing Spotify")
                    return
                isSuccess = self.delegateInput(input)
                print("IsSuccess: {}".format(isSuccess))
                if isSuccess == False:
                    # request failed, un-pause()
                    self.music.pause()
            else:
                self.mic.say("Pardon?")
                #un-pause()
                self.music.pause()


def reconnect(func, *default_args, **default_kwargs):
    """
        Reconnects before running
    """

    def wrap(self, *default_args, **default_kwargs):
        try:
            self.client.connect(self.server, self.port, timeout=8)
        except:
            pass

        # sometimes not enough to just connect
        try:
            return func(self, *default_args, **default_kwargs)
        except:
            self.client = mpd2.MPDClient()
            self.client.timeout = 8
            self.client.idletimeout = None
            self.client.connect(self.server, self.port, timeout=8)

            return func(self, *default_args, **default_kwargs)

    return wrap


class Song(object):
    def __init__(self, id, title, artist, album):

        self.id = id
        self.title = title
        self.artist = artist
        self.album = album


class MPDWrapper(object):
    def __init__(self, server="localhost", port=6600):
        """
            Prepare the client and music variables
        """
        self._logger = logging.getLogger(__name__)

        self.server = server
        self.port = port

        # prepare client
        self.client = mpd2.MPDClient()
        self.client.timeout = 8
        self.client.idletimeout = None
        self.client.connect(self.server, self.port, timeout=8)
        self.client.setvol(50)

        # gather playlists
        #self.playlists = [x["playlist"] for x in self.client.listplaylists()]

        # gather songs
        self.client.clear()
        #for playlist in self.playlists:
        #    self.client.load(playlist)

        #self.songs = []  # may have duplicates
        # capitalized strings
        #self.song_titles = []
        #self.song_artists = []

        #soup = self.client.playlist()
        #for i in range(0, len(soup) / 10):
        #    index = i * 10
        #    id = soup[index].strip()
        #    title = soup[index + 3].strip().upper()
        #    artist = soup[index + 2].strip().upper()
        #    album = soup[index + 4].strip().upper()
        #
        #    self.songs.append(Song(id, title, artist, album))
        #
        #    self.song_titles.append(title)
        #    self.song_artists.append(artist)

    @reconnect
    def playsongbyartist(self, title, artist):
        
        self.client.clear()
        search_results = self.client.search('title', title, 'artist', artist)
        self._logger.debug(search_results[0])
        self.client.add(search_results[0]['file'])
        self.current_song()
        self.play()


    @reconnect
    def rickroll(self):
        search_result = self.client.search('title', 'never gonna give you up', 'artist', 'rick astley')
        
        if search_result:
            search_result = [x for x in search_result if 'track' in x['file']]
            track = search_result[0]
            self._logger.info(track)
            #print(track)
            
            self.client.clear()
            self.client.add(track['file'])
            return (True, track)

        return (False, None)

    @reconnect
    def searchadd(self, query):
        query =query.lower()
        #p = re.compile("play ((?:\w+ ?)+?)by ((?:\w+ ?)+?)(?:on|from) spotify", re.IGNORECASE)
        title_or_artist = "play ((?:songs|music) by )?(?P<title_or_artist>(?:\w+ ?)+?)(?:on|from) spotify"
        title_and_artist = "play (?P<title>(?:\w+ ?)+?)by (?P<artist>(?:\w+ ?)+?)(?:on|from) spotify"
    
        search_result = None
    
        p = re.compile(title_and_artist, re.IGNORECASE)
        match = p.match(query)
    
        list_add = False

        if match and match.group('title').strip() not in ['songs', 'music']:
            self._logger.info(match.groups())
            #print(match.groups())
            title = match.group('title').strip()
            artist = match.group('artist').strip()
            search_result = self.client.search('title', title, 'artist', artist)
        else:
            p = re.compile(title_or_artist, re.IGNORECASE)
            match = p.match(query)
            self._logger.info(match.groups())
            #print(match.groups())
    
            keywords = match.group('title_or_artist').strip()
            if match.group(1) != None:
                if 'music by' in match.group(1) or 'songs by' in match.group(1):
                    search_result = self.client.search('artist', keywords)
                    list_add = True
            else:
                search_result = self.client.search('title', keywords)
        
        if search_result:
            search_result = [x for x in search_result if 'track' in x['file']]
            track = search_result[0]
            self._logger.info(track)
            #print(track)
            
            self.client.clear()
            if list_add:
                [self.client.add(t['file']) for t in search_result]
            else:
                self.client.add(track['file'])
            #self.current_song()
            #self.play()
            return (True, track)
        else:
            return (False, match.groups()) #"Sorry, I couldn't find {} by {}".format(match.groups())



    @reconnect
    def play(self, songs=False, playlist_name=False):
        """
            Plays the current song or accepts a song to play.

            Arguments:
            songs -- a list of song objects
            playlist_name -- user-defined, something like "Love Song Playlist"
        """
        if songs:
            self.client.clear()
            for song in songs:
                try:  # for some reason, certain ids don't work
                    self.client.add(song.id)
                except:
                    pass

        if playlist_name:
            self.client.clear()
            self.client.load(playlist_name)

        if self.state() == 'pause':
            self.client.pause()
        else:
            self.client.play()

    @reconnect
    def current_song(self):
        
        #item = self.client.playlistinfo(int(self.client.status()["song"]))[0]
        #result = "%s by %s" % (item["title"], item["artist"])
        #return result

        status = self.client.status()
        info = self.client.playlistinfo()

        result = ''
        item = None
        print(status)
        print(info)

        index = 0
        if 'song' in status:
            index = int(status['song'])
        elif 'state' in status and 'nextsong' in status:
            index = int(status['nextsong'])
        else:
            return 'No song is currently loaded.'

        item = info[index]    
        print((item['title'], item['artist']))
        result = "%s, by %s" % (item['title'], item['artist'])
        return result

    @reconnect
    def volume(self, level=None, interval=None):

        if level:
            self.client.setvol(int(level))
            return

        if interval:
            level = int(self.client.status()['volume']) + int(interval)
            self.client.setvol(int(level))
            return

    @reconnect
    def pause(self):
        self.client.pause()

    @reconnect
    def stop(self):
        self.client.stop()

    @reconnect
    def next(self):
        self.client.next()
        return

    @reconnect
    def previous(self):
        self.client.previous()
        return

    @reconnect
    def state(self):
        return self.client.status()['state']

    @reconnect
    def seekcur(self, time):
        self.client.seekcur(time)
        return

    def get_soup(self):
        """
        Returns the list of unique words that comprise song and artist titles
        """

        soup = []

        for song in self.songs:
            song_words = song.title.split(" ")
            artist_words = song.artist.split(" ")
            soup.extend(song_words)
            soup.extend(artist_words)

        title_trans = ''.join(chr(c) if chr(c).isupper() or chr(c).islower()
                              else '_' for c in range(256))
        soup = [x.decode('utf-8').encode("ascii", "ignore").upper().translate(
                title_trans).replace("_", "") for x in soup]
        soup = [x for x in soup if x != ""]

        return list(set(soup))

    def get_soup_playlist(self):
        """
        Returns the list of unique words that comprise playlist names
        """

        soup = []

        for name in self.playlists:
            soup.extend(name.split(" "))

        title_trans = ''.join(chr(c) if chr(c).isupper() or chr(c).islower()
                              else '_' for c in range(256))
        soup = [x.decode('utf-8').encode("ascii", "ignore").upper().translate(
                title_trans).replace("_", "") for x in soup]
        soup = [x for x in soup if x != ""]

        return list(set(soup))

    def get_soup_separated(self):
        """
        Returns the list of PHRASES that comprise song and artist titles
        """

        title_soup = [song.title for song in self.songs]
        artist_soup = [song.artist for song in self.songs]

        soup = list(set(title_soup + artist_soup))

        title_trans = ''.join(chr(c) if chr(c).isupper() or chr(c).islower()
                              else '_' for c in range(256))
        soup = [x.decode('utf-8').encode("ascii", "ignore").upper().translate(
                title_trans).replace("_", " ") for x in soup]
        soup = [re.sub(' +', ' ', x) for x in soup if x != ""]

        return soup

    def fuzzy_songs(self, query):
        """
        Returns songs matching a query best as possible on either artist
        field, etc
        """

        query = query.upper()

        matched_song_titles = difflib.get_close_matches(query,
                                                        self.song_titles)
        matched_song_artists = difflib.get_close_matches(query,
                                                         self.song_artists)

        # if query is beautifully matched, then forget about everything else
        strict_priority_title = [x for x in matched_song_titles if x == query]
        strict_priority_artists = [
            x for x in matched_song_artists if x == query]

        if strict_priority_title:
            matched_song_titles = strict_priority_title
        if strict_priority_artists:
            matched_song_artists = strict_priority_artists

        matched_songs_bytitle = [
            song for song in self.songs if song.title in matched_song_titles]
        matched_songs_byartist = [
            song for song in self.songs if song.artist in matched_song_artists]

        matches = list(set(matched_songs_bytitle + matched_songs_byartist))

        return matches

    def fuzzy_playlists(self, query):
        """
                returns playlist names that match query best as possible
        """
        query = query.upper()
        lookup = {n.upper(): n for n in self.playlists}
        results = [lookup[r] for r in difflib.get_close_matches(query, lookup)]
        return resultsA


