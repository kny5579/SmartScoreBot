package org.example.smartScore.util;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class DateUtils {
    
    private static final String DATE_FORMAT = "yyyy-MM-dd";
    private static final SimpleDateFormat DATE_FORMATTER = new SimpleDateFormat(DATE_FORMAT);

    private DateUtils() {
        throw new UnsupportedOperationException("Utility class");
    }

    public static Date parseDate(String dateString) throws ParseException {
        return DATE_FORMATTER.parse(dateString);
    }

    public static String formatDate(Date date) {
        return DATE_FORMATTER.format(date);
    }

    public static SimpleDateFormat getDateFormat() {
        return (SimpleDateFormat) DATE_FORMATTER.clone();
    }
}

